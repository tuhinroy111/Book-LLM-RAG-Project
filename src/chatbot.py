import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load API keys
load_dotenv()

def create_vector_db(pdf_path):
    # 2. Load the PDF
    print(f"Loading document: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 3. Split the text into chunks
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # 4. Create embeddings and FAISS index
    print("Creating embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 5. Save the index locally
    vector_db.save_local("faiss_index")
    print("Done! FAISS index saved to 'faiss_index' folder.")

# Run the function (Replace with your actual PDF filename)
if __name__ == "__main__":
    pdf_filename = "documents/Tuesdays_With_Morrie.pdf" # <-- UPDATE THIS
    create_vector_db(pdf_filename)