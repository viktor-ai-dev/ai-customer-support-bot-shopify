# ai-customer-support

🤖 AI Customer Support (RAG)

An AI-powered customer support system that uses Retrieval-Augmented Generation (RAG) to answer user questions based on uploaded documents such as policies, FAQs, and product data.

Built with a full-stack architecture using FastAPI, Streamlit, and vector search.
🚀 Features

📄 Document-Based AI Support
Upload your own business documents:
-Policies
-FAQs
-Product information
-AI answers questions strictly based on your data

💬 Smart Chat System
-Context-aware conversations with memory
-Automatically rewrites user questions for better retrieval

Uses both:
    Semantic search (vector search)
    Keyword matching (hybrid retrieval)

🧠 RAG (Retrieval-Augmented Generation)
-Splits and embeds documents into a vector database
-Retrieves the most relevant chunks per query
Ensures:
    -Accurate answers
    -No hallucinations (“I don’t know” fallback)

🔐 Authentication System
-User authentication via Supabase
-Each user has their own private document database
-Secure token-based access

💳 Subscription System
-Integrated with Stripe
-Upgrade users to Pro plan
-Webhook automatically updates user access

📚 Source Transparency
-Shows document snippets used to generate answers
-Improves trust and explainability

🏗️ Tech Stack

Backend:

-FastAPI
-Supabase (Auth + Database)
-Stripe (Subscriptions)
-LangChain
-Chroma (Vector Database)
-OpenAI (LLM + embeddings)

Frontend:
-Streamlit
-Requests (API communication)

📂 Project Structure
project/
│── backend.py
│── frontend.py
│── chroma_db/
│── .env

⚙️ How It Works
1. User Authentication
    -User logs in via Supabase
    -Access token is used for secure API calls
2. Document Upload
    -Files are split into chunks
    -Converted into embeddings
    -Stored in a vector database (Chroma)
3. Query Processing
    -User question is rewritten into a standalone query
    -Relevant document chunks are retrieved
4. Hybrid Retrieval
    -Combines semantic search + keyword filtering
    -Ranks and selects the most relevant context
5. AI Response Generation
    -LLM generates an answer using ONLY retrieved context
    -Returns answer + source snippets

🔑 Environment Variables
Create a .env file:

SUPABASE_URL=your_url
SUPABASE_KEY=your_key
STRIPE_SECRET_KEY=your_key
STRIPE_PRICE_ID=your_price_id
STRIPE_WEBHOOK_SECRET=your_webhook_secret

▶️ Run the Project

Backend:
pip install -r requirements.txt
uvicorn backend:app --reload

Frontend:
streamlit run frontend.py

💡 Use Cases
-E-commerce customer support automation
-SaaS support bots
-Internal knowledge base assistants
-FAQ automation systems
-AI chatbots trained on company data

🧠 Key Concepts
-Retrieval-Augmented Generation (RAG)
-Vector search & embeddings
-Hybrid search (semantic + keyword)
-Context-aware chat memory
-Secure multi-user architecture

🔥 Why This Project Matters

This project demonstrates how to build a production-ready AI support system, not just a simple chatbot.

Key highlights:
-Real authentication & user isolation
-Payment integration with subscriptions
-Advanced retrieval pipeline (not basic RAG)
-Context-grounded responses to reduce hallucinations

It showcases skills in:
-Backend API development
-AI system design
-Full-stack integration
-Scalable SaaS architecture

🛠️ Future Improvements:
-Support for PDF and other file formats
-Admin dashboard for analytics
-Conversation history UI
-Fine-tuned models per business
-Multi-language support