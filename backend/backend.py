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
# Score docs
# Check if each word in question exists in doc.
# Give points accordingly.
# --------------------
def score_doc(doc: Document, question: str):
    score = 0
    for word in question.split():
        if word in doc.page_content:
            score += 1
    return score


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

        # chunk_size = antal tecken per text stycke
        # chunk_overlap = efterföljande textstycken delar 100 tecken med föregående chunk(För att inte tappa kontext)
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
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
            metadatas=[{"doc_type":doc_type} for _ in chunks] # Håller reda på vilken doc_type varje chunk har. Funkar som filter.
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
        question = req.question.lower()

        if any(Word in question for Word in ["compare","vs","difference"]):
            filter = {"doc_type":"products"}
        elif any(Word in question for Word in ["return","refund","order","shipping", "delivery", "track", "where"]):
            filter = {"doc_type":"policy"}
        elif any(Word in question for Word in ["warranty","contact","support"]):
            filter = {"doc_type":"faq"}
        else:
            filter = None

        retriever = db.as_retriever(
            search_kwargs={
                "k": 2,
                "filter": filter # Metadata filter
            }
        )

        # Send question(query) and retrieve relevant chunks
        vector_docs = retriever.invoke(req.question)    # Semantic docs = docs där de har enligt AI en relevant betydelse(semantisk)
        keyword_docs = []                               # keyword docs = docs där ett/flera ord finns i question.

        # Loopa igenom chunks(docs), kolla att ett visst ord finns i question.
        for doc in vector_docs:
            if any(Word in doc.page_content.lower() for Word in req.question.lower().split()):
                keyword_docs.append(doc)

        # Merge ALL docs, vector + keyword.
        all_docs = vector_docs + keyword_docs

        unique_docs = []    # unika docs
        seen = set()        # innehållet i docs

        for doc in all_docs:
            if doc.page_content not in seen:    # har vi sätt textinnehållet förut?
                unique_docs.append(doc)         # Spara doc objektet
                seen.add(doc.page_content)      # Spara denna text

        # Top docs:
        # sorted(T@sorted, key) = sorted(list_to_sort, lambda=for each element d in unique_docs: score_doc(d,question)) 
        # returns sorted list in descending order(reverse=True) limit to 5
        top_docs = sorted(unique_docs, key=lambda d: score_doc(d, req.question.lower()),reverse=True)[:5]

        # Join bygger och slår ihop ett set av strängar till en enda sträng, vilket vi bygger med listbyggaren
        # Varje element vi itererar över separeras med \n
        context = "\n".join([doc.page_content for doc in top_docs]) 

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(
            f"""You are a professional ecommerce customer support AI.
                Use ONLY the context below to answer.
                If the answer is not clearly in the context, say 'I don't know'.

                Be helpful, concise and accurate.

                Context:
                {context}

                Question: {req.question}
                
                Answer:
                """
            )

        sources = [doc.page_content[:300] for doc in vector_docs]
        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}