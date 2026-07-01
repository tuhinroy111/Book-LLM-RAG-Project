import json
import logging
from evaluator import RAGEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchEval")

def run_batch_evaluation(client, rag_pipeline_func, dataset_file):
    """
    Loops through the golden dataset and evaluates the RAG pipeline output.
    'rag_pipeline_func' is a function that takes a query and returns (retrieved_chunks, generated_answer).
    'dataset_file' is an uploaded file object (file-like).
    """
    # Load our golden dataset from the file object
    dataset = json.load(dataset_file)

    evaluator = RAGEvaluator(client)
    all_results = []

    logger.info(f"📊 Loaded {len(dataset)} test cases from uploaded dataset.")

    for item in dataset:
        # Extract metadata fields
        test_id = item.get("id", "N/A")
        category = item.get("category", "General")
        query = item.get("user_query")
        expected = item.get("expected_answer_behavior")
        criteria = item.get("pass_fail_criteria")

        # 1. Run your actual RAG pipeline
        # Note: We now correctly unpack the tuple as (answer, chunks)
        # Note: Depending on your specific orchestrator return,
        # ensure retrieved_chunks is the second item.
        generated_answer, retrieved_chunks = rag_pipeline_func(query)

        # 2. Evaluate
        metrics = evaluator.evaluate(
            id=test_id,
            category=category,
            user_query=query,
            expected_behavior=expected,
            pass_fail_criteria=criteria,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer
        )

        all_results.append(metrics)

    # 3. Calculate and log aggregate dataset averages
    total_cases = len(all_results)
    if total_cases == 0:
        logger.warning("No test cases found in JSON.")
        return []

    # (Keep your existing aggregate logging logic here...)
    return all_results