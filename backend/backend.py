from fastapi import FastAPI, UploadFile, File, Form, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
import stripe
import os
from dotenv import load_dotenv
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY or not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID or not STRIPE_WEBHOOK_SECRET:
    raise ValueError("Missing environment variables")

stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-support-frontend-9qcm.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Auth helper --------------------
def get_user_from_token(authorization: str):
    if not authorization:
        raise ValueError("Missing Authorization header")
    token = authorization.replace("Bearer ", "")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.auth.set_session(token, token)
    user = supabase.auth.get_user()
    if not user or not user.user:
        raise ValueError("Invalid user")
    return supabase, user.user.id

# -------------------- Chat model --------------------
class ChatRequest(BaseModel):
    question: str

# -------------------- Create Checkout Session --------------------
@app.post("/create-checkout-session")
async def create_checkout_session(authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url="https://ai-support-frontend-9qcm.onrender.com",
            cancel_url="https://ai-support-frontend-9qcm.onrender.com",
            metadata={"user_id": user_id}
        )
        return JSONResponse(status_code=200, content={"url": session.url})
    except Exception as e:
        print("CREATE CHECKOUT ERROR:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

# -------------------- Stripe Webhook --------------------
@app.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    print("Webhook payload:", payload.decode())
    print("Stripe signature:", sig_header)

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception as e:
        print("Webhook error:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["metadata"]["user_id"]

        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase.table("users_docs").update({"is_pro": True}).eq("user_id", user_id).execute()

        print("🔥 USER UPGRADED:", user_id)

    return JSONResponse(status_code=200, content={"status": "ok"})

# -------------------- Upload File --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_type: str = Form(...), authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)
        content = await file.read()
        text = content.decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks = splitter.split_text(text) or [text]

        embeddings = OpenAIEmbeddings()
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}",
            metadatas=[{"doc_type": doc_type} for _ in chunks]
        )

        supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id,
            "doc_type": doc_type
        }).execute()

        return JSONResponse(status_code=200, content={"status": "uploaded"})
    except Exception as e:
        print("UPLOAD ERROR:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})

# -------------------- Chat Endpoint --------------------
chat_memory = {}

@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)

        if user_id not in chat_memory:
            chat_memory[user_id] = []
        history = chat_memory[user_id]

        history_text = "\n".join([f"User: {h['q']}\nAI: {h['a']}" for h in history[-5:]])

        rewrite_llm = ChatOpenAI(model="gpt-4o-mini")
        rewritten = rewrite_llm.invoke(f"""
        Rewrite the user's question into a standalone question.
        Conversation history:
        {history_text}
        Question:
        {req.question}
        """).content.strip()
        if len(rewritten) < 5:
            rewritten = req.question

        result = supabase.table("users_docs").select("*").eq("user_id", user_id).execute()
        if not result.data:
            return JSONResponse(status_code=400, content={"error": "No documents uploaded"})

        collection_name = result.data[-1]["collection_name"]
        embeddings = OpenAIEmbeddings()
        db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=f"./chroma_db/{collection_name}")
        retriever = db.as_retriever(search_kwargs={"k": 5})

        vector_docs = retriever.invoke(rewritten)
        keyword_docs = [doc for doc in vector_docs if any(word in doc.page_content.lower() for word in rewritten.lower().split())]
        all_docs = vector_docs + keyword_docs

        unique_docs = []
        seen = set()
        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        top_docs = sorted(unique_docs, key=lambda d: sum(1 for w in rewritten.lower().split() if re.search(rf"\b{w}\b", d.page_content.lower())), reverse=True)[:5]
        context = "\n".join([doc.page_content for doc in top_docs])

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(f"""
        You are a professional ecommerce support AI.
        Use ONLY the context below. If not found, say "I don't know".

        Conversation history:
        {history_text}

        Context:
        {context}

        Question:
        {req.question}

        Answer:
        """)
        chat_memory[user_id].append({"q": req.question, "a": response.content})

        sources = []
        seen = set()
        for doc in top_docs:
            snippet = doc.page_content[:150]
            if snippet not in seen:
                sources.append(snippet)
                seen.add(snippet)

        return JSONResponse(status_code=200, content={"answer": response.content, "sources": sources})
    except Exception as e:
        print("CHAT ERROR:", e)
        return JSONResponse(status_code=400, content={"error": str(e)})