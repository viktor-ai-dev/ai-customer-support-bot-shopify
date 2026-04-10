# 🤖 AI Customer Support (RAG)

An AI-powered customer support system that uses Retrieval-Augmented Generation (RAG) to answer user questions based on uploaded documents such as policies, FAQs, and product data.

Built with a full-stack architecture using FastAPI, Streamlit, and vector search.

---

## 🚀 Features

### 📄 Document-Based AI Support

Upload your own business documents:
- Policies  
- FAQs  
- Product information  

AI answers questions strictly based on your data.

---

### 💬 Smart Chat System

- Context-aware conversations with memory  
- Automatically rewrites user questions for better retrieval  

**Uses both:**
- Semantic search (vector search)  
- Keyword matching (hybrid retrieval)  

---

### 🧠 RAG (Retrieval-Augmented Generation)

- Splits and embeds documents into a vector database  
- Retrieves the most relevant chunks per query  

**Ensures:**
- Accurate answers  
- No hallucinations (“I don’t know” fallback)  

---

### 🔐 Authentication System

- User authentication via Supabase  
- Each user has their own private document database  
- Secure token-based access  

---

### 💳 Subscription System

- Integrated with Stripe  
- Upgrade users to Pro plan  
- Webhook automatically updates user access  

---

### 📚 Source Transparency

- Shows document snippets used to generate answers  
- Improves trust and explainability  

---

## 🏗️ Tech Stack

### Backend

- FastAPI  
- Supabase (Auth + Database)  
- Stripe (Subscriptions)  
- LangChain  
- Chroma (Vector Database)  
- OpenAI (LLM + embeddings)  

---

### Frontend

- Streamlit  
- Requests (API communication)  

---

## 📂 Project Structure

```bash
project/
│── backend.py
│── frontend.py
│── chroma_db/
│── .env
