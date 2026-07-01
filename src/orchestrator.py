from langchain_community.vectorstores import FAISS
from indexer import DocumentIndexer
from llm_client import LLMClient
from router import IntentRouter


class RAGOrchestrator:
    def __init__(self, indexer: DocumentIndexer, llm: LLMClient, router: IntentRouter):
        self.indexer = indexer
        self.llm = llm
        self.router = router

    def _contextualize_query(self, question: str, history: list) -> str:
        """Internal helper to convert a follow-up question into a standalone query."""
        if not history:
            return question

        history_text = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in history])
        context_prompt = f"""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that contains all necessary context. Do NOT answer the question, just rephrase it.

        Chat History:
        {history_text}

        Follow-up Question: {question}

        Standalone Question:
        """
        return self.llm.generate_with_retry(context_prompt).strip()

    def run_pipeline(self, question: str, db: FAISS, history: list = None) -> tuple[str, list[str]]:
        """Coordinates guardrails, memory contextualization, database retrieval, and answer synthesis."""
        if history is None:
            history = []

        # 1. Evaluate user intent using modular router
        decision = self.router.route(question)
        if decision.get("intent") == "GREETING":
            return decision.get("direct_response"), []
        if decision.get("intent") == "UNSAFE":
            return decision.get("direct_response"), []

        # 2. Re-write question based on context history
        search_query = self._contextualize_query(question, history)

        # 3. Pull chunks from vector index
        docs = db.similarity_search(search_query, k=10)
        try:
            docs = sorted(docs, key=lambda x: x.metadata.get('page', 0))
        except Exception:
            pass

        # 📄 Injects page metadata directly into the context blocks
        context = "\n---\n".join([f"[Page {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}" for doc in docs])

        # 4. Generate final analytical response
        final_prompt = f"""
            You are an expert analytical AI assistant specialized in deeply reviewing documents.

            Here are the retrieved text snippets from the document:
            {context}

            User Question: {question}

            Instructions:
            1. Provide a comprehensive, detailed, and clear answer using the provided text snippets.
            2. Synthesize context fragments logically.
            3. If the answer is entirely missing from the context fragments, use your reasoning or state clearly what you cannot find.
            4. STRICT CONSTRAINT: Do NOT include any inline citations, page numbers, or parenthetical references (e.g., do NOT write "(Page X)" or "[Page X]") anywhere inside your body text paragraphs. The text must flow cleanly without numbers.

            You MUST format your final response exactly according to this structural layout:

            <Paragraphs of your detailed answer text here, completely free of any page citations>

            Sources Used:
            Page X, Page Y
            """

        generated_answer = self.llm.generate_with_retry(final_prompt)
        retrieved_chunks = [doc.page_content for doc in docs]

        return generated_answer, retrieved_chunks