import streamlit as st
import requests

st.title("AI Document Chat")

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