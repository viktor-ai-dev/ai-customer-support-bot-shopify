import streamlit as st
import requests

# URL till din live backend på Render
BACKEND_URL = "https://ai-customer-backend.onrender.com"

st.set_page_config(page_title="AI Customer Support Chat", page_icon="🤖")
st.title("🤖 AI Customer Support Chat")

# --------------------
# Upload dokument
# --------------------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

file = st.file_uploader("Upload a .txt file", type=["txt"])
if file and st.session_state["user_id"] is None:
    with st.spinner("Uploading..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/upload", files={"file": file})
            resp.raise_for_status()
            user_id = resp.json().get("user_id")
            st.session_state["user_id"] = user_id
            st.success(f"File uploaded! Your session id: {user_id}")
        except Exception as e:
            st.error(f"Failed to upload file: {e}")

# --------------------
# Chat
# --------------------
if st.session_state["user_id"]:
    question = st.chat_input("Ask a question about your document...")
    if question:
        with st.spinner("AI is thinking..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/chat", json={
                    "user_id": st.session_state["user_id"],
                    "question": question
                })
                resp.raise_for_status()
                data = resp.json()
                
                # Visa AI:s svar
                st.chat_message("assistant").write(data.get("answer", "No answer returned"))
                
                # Visa källor
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources used"):
                        for s in sources:
                            st.write(s[:300])
            except Exception as e:
                st.error(f"Failed to get response from backend: {e}")