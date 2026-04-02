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

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ändra till frontend URL i produktion
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_chains = {}

class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text) or [text]

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts( #type:ignore
        chunks, 
        embeddings, 
        collection_name=str(uuid.uuid4()))  #unikt DB varje gång. collection_name=uuid4()
    retriever = db.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_template = (
        "You are a professional customer support AI.\n"
        "Answer ONLY using the context below.\n"
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer:"
    )

    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    qa_chain = RetrievalQA.from_chain_type( #type:ignore
        llm=llm,
        chain_type="stuff",
        retriever=retriever, # each retriever refers to a db with a unique collection name, one for each user
        chain_type_kwargs={"prompt": prompt}
    )

    user_id = str(uuid.uuid4())
    user_chains[user_id] = qa_chain
    return {"user_id": user_id}

@app.post("/chat")
async def chat(req: ChatRequest):
    if req.user_id not in user_chains:
        return {"error": "User not found"}

    qa_chain = user_chains[req.user_id]

    try:
        # Kör chain
        result = qa_chain.invoke({"query": req.question})
        answer = result.get("result", "No answer found")

        # Hämta sources (NYTT SÄTT)
        retriever = qa_chain.retriever
        docs = retriever.invoke(req.question)

        sources = [doc.page_content for doc in docs]

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}