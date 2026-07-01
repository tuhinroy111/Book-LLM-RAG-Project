import logging
import re
from typing import List, Dict

# Configure structured logging for our evaluation pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [RAG-EVAL] - %(message)s'
)
logger = logging.getLogger("RAGEvaluator")


class RAGEvaluator:
    def __init__(self, client):
        """
        Initializes the evaluator with an injected Anthropic client instance.
        """
        self.client = client

    def _parse_score(self, llm_output: str) -> float:
        """Helper to extract a fractional score (e.g., SCORE: 0.75) from LLM text."""
        match = re.search(r"SCORE:\s*([0-9.]+)", llm_output, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Ensure bound between 0.0 and 1.0
            except ValueError:
                pass
        return 0.0

    def evaluate(
            self,
            id: str,
            category: str,
            user_query: str,
            expected_behavior: str,
            pass_fail_criteria: str,
            retrieved_chunks: List[str],
            generated_answer: str
    ) -> Dict[str, float]:
        """
        Runs a comprehensive four-tier evaluation loop for a given query block.
        """
        logger.info(f"🚀 Starting evaluation for TestCase {id} [{category}]")

        precision = self._calculate_precision(id, category, user_query, expected_behavior, pass_fail_criteria,
                                              retrieved_chunks)
        recall = self._calculate_recall(id, category, user_query, expected_behavior, pass_fail_criteria,
                                        retrieved_chunks)
        faithfulness = self._calculate_faithfulness(id, category, user_query, expected_behavior, pass_fail_criteria,
                                                    retrieved_chunks, generated_answer)
        relevancy = self._calculate_relevancy(id, category, user_query, expected_behavior, pass_fail_criteria,
                                              generated_answer)

        metrics = {
            "precision": precision,
            "recall": recall,
            "faithfulness": faithfulness,
            "relevancy": relevancy
        }

        logger.info(f"✅ Completed Evaluation {id} | Scores: {metrics}")
        return metrics

    def _calculate_precision(self, id: str, category: str, query: str, expected: str, criteria: str,
                             retrieved: List[str]) -> float:
        """Calculates the ratio of retrieved chunks that are genuinely relevant."""
        if not retrieved:
            logger.warning(f"[{id}] Precision evaluation skipped: No chunks retrieved.")
            return 0.0

        relevant_count = 0
        for idx, chunk in enumerate(retrieved):
            prompt = f"""You are an expert data analyst evaluating a RAG chunk.
Test ID: {id} | Category: {category}
User Query: {query}
Expected Target Behavior: {expected}
Target Pass/Fail Criteria: {criteria}

Evaluate the following context chunk (Index {idx}):
\"\"\"{chunk}\"\"\"

Task: Does this chunk contain specific information that actively helps satisfy the target behavior or criteria?
Provide a brief reasoning statement, then end your response with strictly: SCORE: 1.0 (if relevant) or SCORE: 0.0 (if completely irrelevant)."""

            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            if "SCORE: 1.0" in response.content[0].text.upper():
                relevant_count += 1

        score = relevant_count / len(retrieved)
        logger.info(f"[{id}] Context Precision: {score:.2f} ({relevant_count}/{len(retrieved)} chunks relevant)")
        return score

    def _calculate_recall(self, id: str, category: str, query: str, expected: str, criteria: str,
                          retrieved: List[str]) -> float:
        """Measures the proportion of expected gold standard information present in context."""
        prompt = f"""You are an expert quality auditor checking RAG context sufficiency.
Test ID: {id} | Category: {category}
User Query: {query}
Expected Answer Behavior: {expected}
Pass/Fail Criteria: {criteria}

Retrieved Context Chunks:
\"\"\"{chr(10).join(retrieved)}\"\"\"

Task: Break down the Expected Answer Behavior and Pass/Fail Criteria into essential facts. What fraction (from 0.0 to 1.0) of those mandatory facts are successfully captured within the Retrieved Context Chunks?
Provide a concise breakdown, then output the final score in this format: SCORE: [float value between 0.0 and 1.0]"""

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        score = self._parse_score(response.content[0].text)
        logger.info(f"[{id}] Context Recall: {score:.2f}")
        return score

    def _calculate_faithfulness(self, id: str, category: str, query: str, expected: str, criteria: str,
                                retrieved: List[str], answer: str) -> float:
        """Determines what percentage of the generated answer is strictly anchored in the context."""
        prompt = f"""You are an expert verification judge auditing an AI response for hallucinations.
Test ID: {id} | Category: {category}
User Query Reference: {query}
Expected Context Guidelines: {expected} | {criteria}

Retrieved Context Chunks (The Source of Truth):
\"\"\"{chr(10).join(retrieved)}\"\"\"

Generated AI Answer:
\"\"\"{answer}\"\"\"

Task: Examine the Generated AI Answer sentence-by-sentence. What proportion (from 0.0 to 1.0) of the claims made in the answer are entirely derived from and supported by the text in the Retrieved Context Chunks? 
Provide your verification steps, then output the final score in this format: SCORE: [float value between 0.0 and 1.0]"""

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        score = self._parse_score(response.content[0].text)
        logger.info(f"[{id}] Answer Faithfulness: {score:.2f}")
        return score

    def _calculate_relevancy(self, id: str, category: str, query: str, expected: str, criteria: str,
                             answer: str) -> float:
        """Scores how effectively the generated answer adheres to target criteria and addresses the user query."""
        prompt = f"""You are an expert grader evaluating alignment.
Test ID: {id} | Category: {category}
User Query: {query}
Expected Answer Behavior: {expected}
Pass/Fail Criteria: {criteria}

Generated AI Answer to Grade:
\"\"\"{answer}\"\"\"

Task: Rate how accurately and thoroughly the Generated AI Answer answers the User Query while strictly complying with the Pass/Fail Criteria. Give a fractional rating from 0.0 (completely missed or failed guardrails) to 1.0 (perfect compliance).
Provide a brief critique, then output the final score in this format: SCORE: [float value between 0.0 and 1.0]"""

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        score = self._parse_score(response.content[0].text)
        logger.info(f"[{id}] Answer Relevancy: {score:.2f}")
        return score