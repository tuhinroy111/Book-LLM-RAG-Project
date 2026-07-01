import streamlit as st
import json
from query import process_uploaded_pdf, ask_bot
from run_eval import run_batch_evaluation
from anthropic import Anthropic
import os

st.set_page_config(page_title="Universal Book RAG", page_icon="📖", layout="wide")

if "messages" not in st.session_state: st.session_state.messages = []
if "db" not in st.session_state: st.session_state.db = None
if "current_file" not in st.session_state: st.session_state.current_file = None

with st.sidebar:
    st.title("Control Panel")
    mode = st.radio("Select Mode", ["Chat", "Evaluation"])

    if mode == "Chat":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and st.session_state.current_file != uploaded_file.name:
            st.session_state.db = process_uploaded_pdf(uploaded_file)
            st.session_state.current_file = uploaded_file.name
            st.success("Indexed!")

    elif mode == "Evaluation":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        json_data = st.file_uploader("Upload Golden JSON", type=["json"])
        if st.button("Run Evaluation"):
            if pdf_file and json_data:
                db = process_uploaded_pdf(pdf_file)
                client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


                def eval_wrapper(query):
                    return ask_bot(query, db)


                results = run_batch_evaluation(client, eval_wrapper, json_data)
                st.write(results)

# Main Chat View
if st.session_state.db:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask a question"):
        with st.chat_message("user"):
            st.markdown(user_query)

        # Unpack the tuple here, ignoring chunks
        answer, _ = ask_bot(user_query, st.session_state.db, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": answer})