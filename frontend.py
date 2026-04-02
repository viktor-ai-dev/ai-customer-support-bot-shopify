import streamlit as st
import requests

st.title("🤖 AI Customer Support Chat")
user_id = st.text_input("Enter your user_id from backend upload:")
question = st.text_input("Your Question?")

if st.button("Ask"):
    if user_id and question:
        resp = requests.post(
            "https://ai-support-bot-0mqr.onrender.com/chat",
            json={"user_id": user_id, "question": question}
        )
        st.write(resp.json())
    else:
        st.warning("Enter user_id and question")

# Upload
file = st.file_uploader("Upload txt file")

if file:
    res = requests.post(
        url="http://127.0.0.1:8000/upload",
        files={"file": file}
    )

    user_id = res.json()["user_id"]
    st.session_state["user_id"] = user_id
    st.success("File uploaded!")

# Chat
if "user_id" in st.session_state:
    question = st.chat_input("Ask something")

    if question:
        res = requests.post(
            url="http://127.0.0.1:8000/chat",
            json={
                "user_id": st.session_state["user_id"],
                "question": question
            }
        )

        data = res.json()
        st.write("🤖", data["answer"])

        with st.expander("Sources"):
            for s in data["sources"]:
                st.write(s[:200])