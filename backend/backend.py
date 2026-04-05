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
import re

load_dotenv()

# Används för Conversation Memory
chat_memory = {}

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
        auth_response.session.access_token, # JWT-token, inte samma som user_id
        auth_response.session.refresh_token
    )
    user_id = str(auth_response.user.id)

    # user_id unikt och sammaför varje användare som loggar in, 
    # eftersom SupaBase kopplar ett unikt, oföränderligt UUID till varje konto.
    return authed_client, user_id 
# --------------------
# Score docs
# Check if each word in question exists in doc.
# Give points accordingly.
# --------------------
def score_doc(doc: Document, question: str):
    words = question.lower().split()
    content = doc.page_content.lower()
    
    # r = raw string, behövs för att använda \b, vilket är en ordgräns.
    # f = f-string, behövs för att kunna ha argument {w} ersätts med ordet
    return sum(1 for w in words if re.search(rf'\b{w}\b', content))

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

        # clear conversional memory if such user don't exist
        if req.user_id not in chat_memory:
            chat_memory[req.user_id] = []

        # history
        history = chat_memory[req.user_id]

        history_text = "\n".join([
            f"User: {h['q']}\nAI: {h['a']}" for h in history[-5:]   # de 5 senaste elementen
        ])

        # Rewrite question (pro feature)
        rewrite_prompt = f"""
        You are an AI assistant.

        Rewrite the user's question into a clear and complete standalone question.
        Use conversation history if needed.

        Conversation history:
        {history_text}

        User question:
        {req.question}

        Rewritten question:
        """

        rewrite_llm = ChatOpenAI(model="gpt-4o-mini")
        rewrite_response = rewrite_llm.invoke(rewrite_prompt)
        rewritten_question = rewrite_response.content.strip()

        # skydd mot dålig rewrite
        if len(rewritten_question) < 5:
            rewritten_question = req.question

        print("ORIGINAL:", req.question)
        print("REWRITTEN:", rewritten_question)

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

        #  --------------------
        # Retriever has a filter for getting chunks with category, for ex: policy.
        # AI Driven Routing here.
        # --------------------
        router_prompt = f"""
        You are an AI router for an ecommerce support system.

        Classify the user's question into ONE of these categories:

        - products → questions about products, comparisons, specifications, features
        - policy → questions about shipping, delivery, returns, refunds, orders, tracking
        - faq → general questions like contact info, support hours, warranty

        IMPORTANT:
        - Questions about shipping, delivery, or orders MUST be classified as "policy"

        ONLY return ONE word: products, policy, or faq.

        Question: {rewritten_question}
        """

        router_llm = ChatOpenAI(model="gpt-4o-mini")
        route = router_llm.invoke(router_prompt).content.strip().lower()     #type: ignore

        # Safeguard
        if route not in ["products","policy","faq"]:
            route = None

        print("QUESTION:", rewritten_question)
        print("AI ROUTE:", route)                                            #type: ignore

        filter = None
        filter = {"doc_type": route} if route in ["products","policy","faq"] else None
      
        retriever = db.as_retriever(
            search_kwargs={
                "k": 2,
                "filter": filter # Metadata filter
            }
        )

        # Send question(query) and retrieve relevant chunks
        # Semantic docs = docs där de har enligt AI en relevant betydelse(semantisk)
        vector_docs = retriever.invoke(rewritten_question)    

        # keyword docs = docs där ett/flera ord finns i question.
        # Loopa igenom chunks(docs), kolla att ett visst ord finns i question.
        keyword_docs = [
          doc for doc in vector_docs if any(Word in doc.page_content.lower() for Word in req.question.lower().split()) 
        ]                               

        # Merge ALL docs, vector + keyword.
        all_docs = vector_docs + keyword_docs

        # dedupe
        unique_docs = []    # unika docs
        seen = set()        # innehållet i docs

        for doc in all_docs:
            if doc.page_content not in seen:    # har vi sätt textinnehållet förut?
                unique_docs.append(doc)         # Spara doc objektet
                seen.add(doc.page_content)      # Spara denna text

        # Top docs
        # returns sorted list in descending order(reverse=True) limit to 5
        top_docs = sorted(unique_docs, key=lambda d: score_doc(doc=d, question=rewritten_question.lower()), reverse=True)[:5]
        
        # Join bygger och slår ihop ett set av strängar till en enda sträng, vilket vi bygger med listbyggaren
        # Varje element vi itererar över separeras med \n
        context = "\n".join([doc.page_content for doc in top_docs]) 

        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(
        f"""
        You are a professional ecommerce support AI.
        Use the conversation history and context below.

        If the answer is not in the context, say "I don't know".

        Conversation history:
        {history_text}

        Context:
        {context}

        Question: {rewritten_question}

        Answer:
        """
        )

        # Vi har fått response, spara conversional memory
        chat_memory[req.user_id].append({"q": rewritten_question, "a": response.content})

        sources = [doc.page_content[:300] for doc in vector_docs]
        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}