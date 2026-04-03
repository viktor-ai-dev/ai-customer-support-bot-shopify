from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

# --------------------
# CORS (frontend access)
# --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-support-frontend-9qcm.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage (RAM)
user_chains = {}

# --------------------
# Request model
# --------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str


# --------------------
# Upload endpoint
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # 1️⃣ Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text) or [text]

        # 2️⃣ Embeddings + Vector DB
        embeddings = OpenAIEmbeddings()

        collection_name = str(uuid.uuid4())  # 🔥 unik per upload

        db = Chroma.from_texts(  # type: ignore
            texts=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )

        retriever = db.as_retriever(search_kwargs={"k": 2})

        # 3️⃣ LLM + prompt
        llm = ChatOpenAI(model="gpt-4o-mini")

        prompt_template = (
            "You are a professional customer support AI.\n"
            "Answer ONLY using the context below.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer:"
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # 4️⃣ Chain
        qa_chain = RetrievalQA.from_chain_type(  # type: ignore
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )

        # 5️⃣ Create user session
        user_id = str(uuid.uuid4())
        user_chains[user_id] = qa_chain

        return {"user_id": user_id}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}


# --------------------
# Chat endpoint
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    if req.user_id not in user_chains:
        return {"error": "User not found (session expired or invalid ID)"}

    qa_chain = user_chains[req.user_id]

    try:
        # 1️⃣ Run chain (NEW LangChain API)
        result = qa_chain.invoke({"query": req.question})
        answer = result.get("result", "I couldn't find an answer.")

        # 2️⃣ Get sources (NEW retriever API)
        retriever = qa_chain.retriever
        docs = retriever.invoke(req.question)

        sources = [doc.page_content[:300] for doc in docs]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}