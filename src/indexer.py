import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker


class DocumentIndexer:
    def __init__(self):
        # Local open-source model running entirely on your machine
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def index_pdf(self, uploaded_file) -> FAISS:
        """Saves a streamlit stream, breaks it down semantically, and indexes it."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Semantic chunker clusters lines by contextual meaning
            text_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
            docs = text_splitter.split_documents(documents)

            return FAISS.from_documents(docs, self.embeddings)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)