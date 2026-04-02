from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
import os
from dotenv import load_dotenv

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

# Temp storage för användarens chains
user_chains = {}

# --------------------
# Request Model
# --------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str

# --------------------
# Upload endpoint
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(text) or [text]

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(texts=chunks, embedding=embeddings) #type: ignore
    retriever = db.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt_template = (
        "You are a professional customer support AI. "
        "Answer ONLY using the context below. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type( #type: ignore
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    user_id = str(uuid.uuid4())
    user_chains[user_id] = qa_chain

    return {"user_id": user_id}

# --------------------
# Chat endpoint
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):                           #type: ignore
    if req.user_id not in user_chains:
        return {"error": "User not found"}

    qa_chain = user_chains[req.user_id]                     #type: ignore
    result = qa_chain.run(req.question)                     #type: ignore

    retriever = qa_chain.retriever                          #type: ignore
    docs = retriever.get_relevant_documents(req.question)   #type: ignore
    sources = [doc.page_content for doc in docs]            #type: ignore

    return {"answer": result, "sources": sources}           #type: ignore

# --------------------
# Root endpoint
# --------------------
@app.get("/")
async def root():
    return {"message": "AI Customer Support Bot is live!"}