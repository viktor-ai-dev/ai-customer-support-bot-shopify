import streamlit as st
import requests
import os

st.set_page_config(page_title="🤖 AI Customer Support Chat", page_icon="🤖")
st.title("🤖 AI Customer Support Chat")

# Läs backend URL från miljövariabel
BACKEND_URL = os.getenv("BACKEND_URL", "https://ai-support-bot-0mqr.onrender.com")

# ------------------------
# Upload
# ------------------------
st.header("📄 Upload your support document")
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    with st.spinner("Uploading file..."):
        res = requests.post(
            url=f"{BACKEND_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "text/plain")}
        )
    if res.status_code == 200:
        user_id = res.json()["user_id"]
        st.session_state["user_id"] = user_id
        st.success(f"File uploaded! Your session ID: {user_id}")
    else:
        st.error("Failed to upload file")

# ------------------------
# Chat
# ------------------------
st.header("💬 Ask questions")
if "user_id" not in st.session_state:
    st.info("Upload a file first to get a session ID.")
else:
    question = st.text_input("Your question:")

    if st.button("Ask") and question:
        with st.spinner("AI is thinking..."):
            res = requests.post(
                url=f"{BACKEND_URL}/chat",
                json={
                    "user_id": st.session_state["user_id"],
                    "question": question
                }
            )
        if res.status_code == 200:
            data = res.json()
            st.write("🤖", data.get("answer", "No answer"))
            sources = data.get("sources", [])
            if sources:
                with st.expander("🔍 Sources"):
                    for i, s in enumerate(sources, 1):
                        st.write(f"{i}. {s[:300]}...")
        else:
            st.error("Failed to get response from backend")