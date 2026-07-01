from indexer import DocumentIndexer
from llm_client import LLMClient
from router import IntentRouter
from orchestrator import RAGOrchestrator

# Initialize our singletons
_indexer = DocumentIndexer()
_llm = LLMClient()
_router = IntentRouter()

# Wire them together into the Orchestrator
_orchestrator = RAGOrchestrator(_indexer, _llm, _router)

def process_uploaded_pdf(uploaded_file):
    return _indexer.index_pdf(uploaded_file)

def ask_bot(question, db, history=None):
    if history is None:
        history = []
    # Return the full tuple (answer, chunks)
    return _orchestrator.run_pipeline(question, db, history)