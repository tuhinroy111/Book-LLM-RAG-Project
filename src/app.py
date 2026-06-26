import streamlit as st
from query import process_uploaded_pdf, ask_bot

st.set_page_config(page_title="Universal Book RAG", page_icon="📖", layout="wide")

# Persistent storage configurations
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db" not in st.session_state:
    st.session_state.db = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar layout for file handling
with st.sidebar:
    st.title("⚙️ Document Control")
    uploaded_file = st.file_uploader("Upload any book or document (PDF)", type=["pdf"])

    if uploaded_file:
        # If a brand new file is uploaded, wipe old history and re-index
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.db = None
            st.session_state.current_file = uploaded_file.name

            with st.spinner("Analyzing document structure via semantic chunking..."):
                try:
                    st.session_state.db = process_uploaded_pdf(uploaded_file)
                    st.success(f"Successfully indexed: {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

# Main Chat View
st.title("📖 Dynamic Universal RAG Chatbot")
st.write("Upload a document in the sidebar to start a context-aware chat conversation.")

if st.session_state.db is None:
    st.info("👈 Please upload a PDF book in the left sidebar to begin.")
else:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat interface input
    if user_query := st.chat_input(f"Ask anything about {st.session_state.current_file}..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Scanning semantic layers and generating response..."):
                try:
                    # Execute bot search with session database
                    answer = ask_bot(user_query, st.session_state.db, history=st.session_state.messages)
                    response_placeholder.markdown(answer)
                except Exception as e:
                    response_placeholder.error(f"An error occurred: {e}")
                    answer = "An error occurred while fetching the answer."

        # Commit to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": answer})