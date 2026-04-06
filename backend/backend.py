from fastapi import FastAPI, UploadFile, File, Form, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import stripe
from fastapi import Header


load_dotenv()

# --------------------
# ENV
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

# --------------------
# Memory
# --------------------
chat_memory = {}

# --------------------
# FastAPI
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-support-frontend-9qcm.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Request model
# --------------------
class ChatRequest(BaseModel):
    question: str

# --------------------
# Auth helper (JWT)
# --------------------
def get_user_from_token(authorization: str):
    if not authorization:
        raise ValueError("Missing Authorization header")

    token = authorization.replace("Bearer ", "")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # sätt user session (viktigt för RLS)
    supabase.auth.set_session(token, token)

    user = supabase.auth.get_user()

    if not user or not user.user:
        raise ValueError("Invalid user")

    return supabase, user.user.id

# --------------------
# Score docs (keyword ranking)
# --------------------
def score_doc(doc, question: str):
    words = question.lower().split()
    content = doc.page_content.lower()
    return sum(1 for w in words if re.search(rf"\b{w}\b", content))

# --------------------
# Extract clean snippet
# --------------------
def extract_relevant_snippet(doc, question):
    sentences = doc.page_content.split(".")
    
    for s in sentences:
        if any(word in s.lower() for word in question.lower().split()):
            return s.strip()
    
    return doc.page_content[:150]

# --------------------
# Create Checkout Session
# --------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.post("/create-checkout-session")
async def create_checkout_session(authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{
                "price": os.getenv("STRIPE_PRICE_ID"),
                "quantity": 1,
            }],
            success_url="https://ai-support-frontend-9qcm.onrender.com",
            cancel_url="https://ai-support-frontend-9qcm.onrender.com",
            metadata={
                "user_id": user_id
            }
        )

        return {"url": session.url}

    except Exception as e:
        return {"error": str(e)}
    
# --------------------
# Webhook
# --------------------
@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    print("Webhook payload:", payload.decode())
    print("Stripe signature:", sig_header)
    
    try:
        event = stripe.Webhook.construct_event(
            payload,
            sig_header,
            os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except Exception as e:
        print("Webhook error:", e)
        return {"error": str(e)}

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        user_id = session["metadata"]["user_id"]

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        supabase.table("users_docs").update({
            "is_pro": True
        }).eq("user_id", user_id).execute()

        print("🔥 USER UPGRADED:", user_id)

    return {"status": "ok"}

# --------------------
# UPLOAD
# --------------------
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    authorization: str = Header(None)
):
    try:
        supabase, user_id = get_user_from_token(authorization)

        content = await file.read()
        text = content.decode("utf-8")

        # bättre chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )
        chunks = splitter.split_text(text) or [text]

        embeddings = OpenAIEmbeddings()

        # skapa / uppdatera vector db
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}",
            metadatas=[{"doc_type": doc_type} for _ in chunks]
        )

        # spara i supabase
        supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id,
            "doc_type": doc_type
        }).execute()

        return {"status": "uploaded"}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}

# --------------------
# CHAT
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)

        # --------------------
        # Memory
        # --------------------
        if user_id not in chat_memory:
            chat_memory[user_id] = []

        history = chat_memory[user_id]

        history_text = "\n".join([
            f"User: {h['q']}\nAI: {h['a']}"
            for h in history[-5:]
        ])

        # --------------------
        # Rewrite question
        # --------------------
        rewrite_llm = ChatOpenAI(model="gpt-4o-mini")

        rewritten = rewrite_llm.invoke(f"""
        Rewrite the user's question into a clear standalone question.

        Conversation history:
        {history_text}

        Question:
        {req.question}
        """).content.strip()

        if len(rewritten) < 5:
            rewritten = req.question

        # --------------------
        # Get user DB
        # --------------------
        result = supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()

        if not result.data:
            return {"error": "No documents uploaded"}

        collection_name = result.data[-1]["collection_name"]

        embeddings = OpenAIEmbeddings()

        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        # --------------------
        # Retriever
        # --------------------
        retriever = db.as_retriever(search_kwargs={"k": 5})

        vector_docs = retriever.invoke(rewritten)

        # keyword boost
        keyword_docs = [
            doc for doc in vector_docs
            if any(word in doc.page_content.lower() for word in rewritten.lower().split())
        ]

        all_docs = vector_docs + keyword_docs

        # dedupe
        unique_docs = []
        seen = set()

        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        # rank
        top_docs = sorted(
            unique_docs,
            key=lambda d: score_doc(d, rewritten),
            reverse=True
        )[:5]

        # context
        context = "\n".join([doc.page_content for doc in top_docs])

        # --------------------
        # Final LLM
        # --------------------
        llm = ChatOpenAI(model="gpt-4o-mini")

        response = llm.invoke(f"""
        You are a professional ecommerce support AI.

        Use ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Conversation history:
        {history_text}

        Context:
        {context}

        Question:
        {req.question}

        Answer:
        """)

        # save memory
        chat_memory[user_id].append({
            "q": req.question,
            "a": response.content
        })

        # --------------------
        # Clean sources
        # --------------------
        sources = []
        seen = set()

        for doc in top_docs:
            snippet = extract_relevant_snippet(doc, rewritten)

            if snippet not in seen:
                sources.append(snippet)
                seen.add(snippet)

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}
