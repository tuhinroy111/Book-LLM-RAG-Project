import streamlit as st
import json
import os
from query import process_uploaded_pdf, ask_bot
from run_eval import run_batch_evaluation
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Universal Book RAG", page_icon="📖", layout="wide")


# 🔄 Function to wipe out previous files and chat history when switching modes
def reset_context():
    st.session_state.messages = []
    st.session_state.db = None
    st.session_state.current_file = None


# 🧠 Persistent storage initialization
if "messages" not in st.session_state: st.session_state.messages = []
if "db" not in st.session_state: st.session_state.db = None
if "current_file" not in st.session_state: st.session_state.current_file = None

with st.sidebar:
    st.title("⚙️ Control Panel")
    # The on_change parameter triggers our reset function right when a toggle happens
    mode = st.radio("Select Mode", ["Chat 💬", "Evaluation 📊"], on_change=reset_context)

    if mode == "Chat 💬":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and st.session_state.current_file != uploaded_file.name:
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                st.session_state.db = process_uploaded_pdf(uploaded_file)
                st.session_state.current_file = uploaded_file.name
            st.success("Indexed successfully!")

    elif mode == "Evaluation 📊":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        json_data = st.file_uploader("Upload Golden JSON", type=["json"])
        eval_button = st.button("Run Evaluation")

# 📐 Define Layout: Main View (ratio 3) and Right Panel (ratio 1)
main_col, right_col = st.columns([3, 1])

# --- 📋 Right Panel (System Status) ---
with right_col:
    st.subheader("ℹ️ System Status")
    if st.session_state.current_file:
        st.success(f"**Active DB:**\n{st.session_state.current_file}")
    else:
        st.warning("No document loaded.")

    st.markdown("---")
    st.markdown(
        "**Instructions:**\n* **Chat:** Upload a PDF to start asking questions.\n* **Evaluation:** Upload both a PDF and a Golden JSON dataset to calculate metrics.")

# --- 💻 Main View ---
with main_col:
    if mode == "Chat 💬":
        st.title("💬 Document Chat")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Disable input if no file is uploaded yet
        input_disabled = st.session_state.db is None
        placeholder_text = "Ask a question..." if not input_disabled else "Upload a PDF in the Control Panel first."

        if user_query := st.chat_input(placeholder=placeholder_text, disabled=input_disabled):
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Analyzing document and generating answer..."):
                answer, _ = ask_bot(user_query, st.session_state.db, st.session_state.messages)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": answer})

    elif mode == "Evaluation 📊":
        st.title("📊 Pipeline Evaluation")

        if eval_button:
            if pdf_file and json_data:
                with st.status("Starting Evaluation Pipeline...", expanded=True) as status:
                    st.write("1️⃣ Indexing PDF document...")
                    db = process_uploaded_pdf(pdf_file)
                    st.session_state.current_file = pdf_file.name

                    st.write("2️⃣ Initializing LLM Client...")
                    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


                    def eval_wrapper(query):
                        return ask_bot(query, db)


                    st.write("3️⃣ Running batch evaluation against JSON dataset...")
                    results = run_batch_evaluation(client, eval_wrapper, json_data)

                    status.update(label="Evaluation Complete!", state="complete", expanded=False)

                if results:
                    st.success("All metrics calculated successfully.")

                    # Calculate aggregates
                    avg_precision = sum(r.get("precision", 0) for r in results) / len(results)
                    avg_recall = sum(r.get("recall", 0) for r in results) / len(results)
                    avg_faithfulness = sum(r.get("faithfulness", 0) for r in results) / len(results)
                    avg_relevancy = sum(r.get("relevancy", 0) for r in results) / len(results)

                    # Display clean metric cards
                    st.subheader("🎯 Aggregate Scores")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(label="Context Precision", value=f"{avg_precision:.2f}")
                    col2.metric(label="Context Recall", value=f"{avg_recall:.2f}")
                    col3.metric(label="Faithfulness", value=f"{avg_faithfulness:.2f}")
                    col4.metric(label="Answer Relevancy", value=f"{avg_relevancy:.2f}")

                    # Detailed view in an expander
                    with st.expander("🔍 View Detailed Test Cases"):
                        st.dataframe(results, use_container_width=True)
                else:
                    st.warning("No results were returned. Check your JSON formatting.")
            else:
                st.error("Please upload both the PDF and JSON files in the Control Panel.")
        else:
            st.info("Upload your files and click 'Run Evaluation' to begin.")