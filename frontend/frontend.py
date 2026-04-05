import streamlit as st
import requests
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------
# Konstanter
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

# ✅ skapa EN gång
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BACKEND_URL = "https://ai-customer-backend.onrender.com"

st.set_page_config(page_title="AI Customer Support Chat", page_icon="🤖")
st.title("🤖 AI Customer Support Chat")

# --------------------
# SESSION INIT
# --------------------
if "user" not in st.session_state:
    st.session_state["user"] = None

# --------------------
# LOGIN / SIGNUP
# --------------------
if not st.session_state["user"]:
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                st.session_state["user"] = res.user
                st.success("Logged in!")
                st.rerun()

            except Exception as e:
                st.error(f"Login failed: {e}")

    with col2:
        if st.button("Sign Up"):
            try:
                res = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })

                st.success("Account created! Please log in.")

            except Exception as e:
                st.error(f"Signup failed: {e}")

# --------------------
# APP (efter login)
# --------------------
if st.session_state["user"]:

    user_id = st.session_state["user"].id
    st.success(f"Logged in as: {st.session_state['user'].email}")

    # --------------------
    # Document type
    # --------------------
    doc_type = st.selectbox(
        "Select document type",
        ["policy", "products", "faq"]
    )

    # --------------------
    # Upload
    # --------------------
    file = st.file_uploader("Upload a .txt file", type=["txt"])

    if file:
        with st.spinner("Uploading..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": file},
                    data={
                        "user_id": user_id,
                        "doc_type": doc_type
                    }
                )

                resp.raise_for_status()
                st.success("File uploaded!")

            except Exception as e:
                st.error(f"Upload failed: {e}")

    # --------------------
    # Chat
    # --------------------
    question = st.chat_input("Ask a question...")

    if question:
        with st.spinner("AI is thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "user_id": user_id,
                        "question": question
                    }
                )

                resp.raise_for_status()
                data = resp.json()

                st.chat_message("assistant").write(
                    data.get("answer", "No answer returned")
                )

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            st.write(s[:300])

            except Exception as e:
                st.error(f"Chat failed: {e}")