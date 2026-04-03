from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

load_dotenv()

# --------------------
# Supabase (global klient, används bara för inloggning)
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Måste vara anon key i produktion

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --------------------
# App
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
# Upload
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text) or [text]
        embeddings = OpenAIEmbeddings()

        # --------------------
        # Autentisering med anon key
        # --------------------
        supabase_email = os.getenv("SUPABASE_USERNAME")
        supabase_pass = os.getenv("SUPABASE_PASSWORD")

        if not supabase_email or not supabase_pass:
            raise ValueError("Supabase email/password missing in .env")

        # Logga in med den globala klienten
        auth_response = supabase.auth.sign_in_with_password({
            "email": supabase_email,
            "password": supabase_pass
        })

        if not auth_response.session:
            raise ValueError("Login failed - no session returned")

        # Skapa en NY klient enbart för denna användare, sätt token manuellt
        authed_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        authed_supabase.auth.set_session(
            auth_response.session.access_token,
            auth_response.session.refresh_token
        )

        user_id = str(auth_response.user.id)
        print("Inloggad användare:", user_id)

        # --------------------
        # Skapa Chroma-databas
        # --------------------
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}"
        )

        # --------------------
        # Spara i Supabase med den autentiserade klienten
        # --------------------
        result = authed_supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id
        }).execute()

        print("Insert result:", result.data)

        return {"user_id": user_id}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}

# --------------------
# Chat
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print("Incoming request: ", req)

        # Här räcker det med den globala klienten, eftersom vi bara läser data
        # (SELECT kräver RLS-policy, men vi har redan en SELECT-policy för användaren)
        response = supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .execute()
        
        print("Supabase response:", response.data)

        if not response.data:
            return {"error": "User not found"}

        collection_name = response.data[0]["collection_name"]

        embeddings = OpenAIEmbeddings()
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        retriever = db.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(req.question)
        context = "\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(
            f"""You are a professional customer support AI.
                Answer ONLY using the context below.
                If the answer is not in the context, say 'I don't know'.

                Context:
                {context}

                Question: {req.question}
                Answer:"""
            )

        sources = [doc.page_content[:300] for doc in docs]
        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}