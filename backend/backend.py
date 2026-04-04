from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from fastapi import Form

load_dotenv()

# --------------------
# Konstanter (ingen global klient)
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")          # anon key
SUPABASE_EMAIL = os.getenv("SUPABASE_USERNAME")
SUPABASE_PASS = os.getenv("SUPABASE_PASSWORD")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")
if not SUPABASE_EMAIL or not SUPABASE_PASS:
    raise ValueError("Missing Supabase email/password")

# --------------------
# Hjälpfunktion: skapar en autentiserad Supabase-klient lokalt
# --------------------
def get_authed_supabase() -> tuple[Client, str]:

    # Temporär oautentiserad klient för inloggning
    # Returnerar JWT-token.
    temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    auth_response = temp_client.auth.sign_in_with_password({
        "email": SUPABASE_EMAIL,
        "password": SUPABASE_PASS
    })

    if not auth_response.session:
        raise ValueError("Login failed - no session returned")
    
    # Skapa en ny autentiserad klient med samma nyckel
    authed_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Använder authed_response från temp_client med giltig JWT-token.
    authed_client.auth.set_session(
        auth_response.session.access_token,
        auth_response.session.refresh_token
    )
    user_id = str(auth_response.user.id)
    return authed_client, user_id

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
async def upload(file: UploadFile = File(...), 
                 doc_type: str = Form(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text) or [text]
        embeddings = OpenAIEmbeddings()

        # Skapa lokal autentiserad klient
        authed_supabase, user_id = get_authed_supabase()
        print("Inloggad användare:", user_id)

        # Skapa Chroma-databas
        Chroma.from_texts(
            texts=chunks,       # chunks sparas i databas
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}",
            metadatas=[{"doc_type":doc_type} for _ in chunks] # Håller reda på vilken doc_type varje chunk har
        )

        # Spara i Supabase
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
        print("Incoming request:", req)

        # Skapa lokal autentiserad klient
        authed_supabase, logged_in_user_id = get_authed_supabase()
        print("Inloggad användare (RLS):", logged_in_user_id)

        # Hämta användarens dokument (RLS filtrerar automatiskt på logged_in_user_id)
        result = authed_supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .execute()
        
        print("Supabase response:", result.data)

        if not result.data:
            return {"error": "User not found"}

        # From database query result, we get the collection_name
        # Accessing the first JSON objects content, column name: collection_name which is actually the user_id
        collection_name = result.data[0]["collection_name"]

        embeddings = OpenAIEmbeddings()
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        # Retriever has a filter for getting chunks with category, for ex: policy.
        # TESTING NOW: Always filter on chunk category: Policy
        retriever = db.as_retriever(
            search_kwargs={
                "k": 2,
                #"filter": {"doc_type": "policy"} # Metadata filter
            }
        )

        # Send question(query) and retrieve relevant chunks
        docs = retriever.invoke(req.question)

        # Join bygger och slår ihop ett set av strängar till en enda sträng, vilket vi bygger med listbyggaren
        # Varje element vi itererar över separeras med \n
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