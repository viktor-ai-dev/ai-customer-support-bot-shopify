from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

load_dotenv()

# --------------------
# ENV
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

# ✅ Single global client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------
# Memory (per user)
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
    user_id: str
    question: str

# --------------------
# Score function (keyword ranking)
# --------------------
def score_doc(doc, question: str):
    words = question.lower().split()
    content = doc.page_content.lower()
    return sum(1 for w in words if re.search(rf"\b{w}\b", content))

# --------------------
# Upload endpoint
# --------------------
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    doc_type: str = Form(...)
):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_text(text) or [text]

        embeddings = OpenAIEmbeddings()

        # Create vector DB (per user)
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}",
            metadatas=[{"doc_type": doc_type} for _ in chunks]
        )

        # Save mapping in Supabase
        supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id,
            "doc_type": doc_type
        }).execute()

        return {"user_id": user_id}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}

# --------------------
# Chat endpoint
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print("Incoming:", req)

        # --------------------
        # Memory
        # --------------------
        if req.user_id not in chat_memory:
            chat_memory[req.user_id] = []

        history = chat_memory[req.user_id]

        history_text = "\n".join([
            f"User: {h['q']}\nAI: {h['a']}"
            for h in history[-5:]
        ])

        # --------------------
        # Query Rewriting
        # --------------------
        rewrite_llm = ChatOpenAI(model="gpt-4o-mini")

        rewrite_prompt = f"""
        Rewrite the user's question into a clear standalone question.

        Conversation history:
        {history_text}

        User question:
        {req.question}

        Rewritten question:
        """

        rewritten = rewrite_llm.invoke(rewrite_prompt).content.strip()

        if len(rewritten) < 5:
            rewritten = req.question

        print("ORIGINAL:", req.question)
        print("REWRITTEN:", rewritten)

        # --------------------
        # Get user collection
        # --------------------
        result = supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .execute()

        if not result.data:
            return {"error": "User not found"}

        collection_name = result.data[-1]["collection_name"]

        # --------------------
        # Load DB
        # --------------------
        embeddings = OpenAIEmbeddings()

        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        # --------------------
        # AI Routing
        # --------------------
        router_llm = ChatOpenAI(model="gpt-4o-mini")

        router_prompt = f"""
        You are an ecommerce support classifier.

        Classify into:
        - products → product info, specs, comparisons
        - policy → shipping, delivery, orders, tracking, returns
        - faq → general questions (contact, hours, warranty)

        IMPORTANT:
        - Questions about orders MUST be "policy"
        - Questions about shipping MUST be "policy"

        Question: {rewritten}

        Answer with ONE word only.
        """

        route = router_llm.invoke(router_prompt).content.strip().lower()

        if route not in ["products", "policy", "faq"]:
            route = None

        print("ROUTE:", route)

        # --------------------
        # Retriever
        # --------------------
        retriever = db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"doc_type": route} if route else None
            }
        )

        # Vector search
        vector_docs = retriever.invoke(rewritten)

        # Keyword search
        keyword_docs = [
            doc for doc in vector_docs
            if any(word in doc.page_content.lower() for word in rewritten.lower().split())
        ]

        # Merge
        all_docs = vector_docs + keyword_docs

        # Deduplicate
        unique_docs = []
        seen = set()

        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        # Rank
        top_docs = sorted(
            unique_docs,
            key=lambda d: score_doc(d, rewritten),
            reverse=True
        )[:5]

        # Build context
        context = "\n".join([doc.page_content for doc in top_docs])

        # Debugging
        print("DOCS FOUND:", len(vector_docs))
        print("CONTEXT:", context[:200])

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

        Question: {req.question}

        Answer:
        """)

        # Save memory
        chat_memory[req.user_id].append({
            "q": req.question,
            "a": response.content
        })

        # Sources
        sources = [doc.page_content[:300] for doc in top_docs]

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}
