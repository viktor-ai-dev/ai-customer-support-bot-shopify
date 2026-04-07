import streamlit as st
import requests
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BACKEND_URL = "https://ai-customer-backend.onrender.com"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AI Customer Support Chat", page_icon="🤖")
st.title("🤖 AI Customer Support Chat")

if "user" not in st.session_state: st.session_state["user"] = None
if "access_token" not in st.session_state: st.session_state["access_token"] = None

# -------------------- Upgrade Button --------------------
if st.button("🚀 Upgrade to Pro"):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/create-checkout-session",
            headers={"Authorization": f"Bearer {st.session_state['access_token']}"}
        )
        try:
            data = resp.json()
        except Exception:
            st.error(f"Backend returned non-JSON:\n{resp.text}")
            st.stop()
        if resp.status_code != 200:
            st.error(f"Error creating checkout:\n{data.get('error')}")
        else:
            url = data.get("url")
            if url:
                st.markdown(f"[👉 Pay here]({url})")
            else:
                st.error("No URL returned from backend")
    except Exception as e:
        st.error(f"Request failed: {e}")

# -------------------- Login / Signup --------------------
if not st.session_state["user"]:
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state["user"] = res.user
                st.session_state["access_token"] = res.session.access_token
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    with col2:
        if st.button("Sign Up"):
            try:
                supabase.auth.sign_up({"email": email, "password": password})
                st.success("Account created! Please log in.")
            except Exception as e:
                st.error(f"Signup failed: {e}")

# -------------------- Main App --------------------
if st.session_state["user"]:
    st.success(f"Logged in as: {st.session_state['user'].email}")
    headers = {"Authorization": f"Bearer {st.session_state['access_token']}"}

    doc_type = st.selectbox("Select document type", ["policy", "products", "faq"])
    file = st.file_uploader("Upload a .txt file", type=["txt"])
    if file:
        try:
            with st.spinner(text="Processing..."):
                resp = requests.post(f"{BACKEND_URL}/upload", files={"file": file}, data={"doc_type": doc_type}, headers=headers)
                try:
                    data = resp.json()
                except Exception:
                    st.error(f"Backend returned non-JSON:\n{resp.text}")
                    st.stop()
                if resp.status_code != 200:
                    st.error(f"Upload failed: {data.get('error')}")
                else:
                    st.success("File uploaded!")
        except Exception as e:
            st.error(f"Upload request failed: {e}")

    question = st.chat_input("Ask a question...")
    if question:
        try:
            resp = requests.post(f"{BACKEND_URL}/chat", json={"question": question}, headers=headers)
            try:
                data = resp.json()
            except Exception:
                st.error(f"Backend returned non-JSON:\n{resp.text}")
                st.stop()
            if resp.status_code != 200:
                st.error(f"Chat failed: {data.get('error')}")
            else:
                st.chat_message("assistant").write(data.get("answer", "No answer"))
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            st.write(s[:300])
        except Exception as e:
            st.error(f"Chat request failed: {e}")