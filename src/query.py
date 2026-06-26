import os
import tempfile
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Initialize embeddings once globally
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_uploaded_pdf(uploaded_file):
    """
    Takes an uploaded file from Streamlit, saves it temporarily,
    splits it using Semantic Chunking, and returns a FAISS vector database.
    """
    # 1. Save uploaded file to a temporary location to load it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 3. HIGH PRECISION: Semantic Chunking
        # This splits text based on semantic shifts between sentences rather than character counts.
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        docs = text_splitter.split_documents(documents)

        # 4. Create an in-memory FAISS database
        db = FAISS.from_documents(docs, embeddings)
        return db
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ask_bot(question, db, history=[]):
    """
    Accepts the query, active database instance, and chat history.
    """
    # 1. Contextualize the question based on chat history
    search_query = question
    if len(history) > 0:
        history_text = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in history])

        context_prompt = f"""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that contains all necessary context. Do NOT answer the question, just rephrase it.

        Chat History:
        {history_text}

        Follow-up Question: {question}

        Standalone Question:
        """
        rewrite_response = client.models.generate_content(model='gemini-2.5-flash', contents=context_prompt)
        search_query = rewrite_response.text.strip()

    # 2. Search our dynamic database instance
    docs = db.similarity_search(search_query, k=10)

    try:
        docs = sorted(docs, key=lambda x: x.metadata.get('page', 0))
    except Exception:
        pass

    context = "\n---\n".join([doc.page_content for doc in docs])

    # 3. Generate Answer
    final_prompt = f"""
    You are an expert analytical AI assistant specialized in deeply reviewing documents.

    Here are the retrieved text snippets from the document:
    {context}

    User Question: {question}

    Instructions:
    1. Provide a comprehensive, detailed, and clear answer using the provided text snippets.
    2. Synthesize context fragments logically. If the user asks for a summary or structural overview, connect the dots across sections cleanly.
    3. If the answer is entirely missing from the context fragments, use your reasoning or state clearly what you cannot find.

    Detailed Answer:
    """
    response = client.models.generate_content(model='gemini-2.5-flash', contents=final_prompt)
    return response.text