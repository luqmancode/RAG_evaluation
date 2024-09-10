""" RAG Evaluation """
import logging

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample data
documents = {
    1: "A busy bridge in northern Vietnam collapsed after being hit by Super Typhoon Yagi, which has killed more than 60 people since making landfall on Saturday.",
    2: "Flintoff named England Lions head coach",
    3: "Supreme Court slams Kolkata police for delay in registering FIR; doctors are told to join work",
}

# Retrieved documents for queries
retrieval_results = {"query1": [1, 3], "query2": [2, 3]}

# Ground truth documents
ground_truth_retrieval = {"query1": [1], "query2": [2]}


# Retrieval Evaluation: Recall@K
def evaluate_retrieval(retrieval_results, ground_truth, k=2) -> float:
    """
    Evaluates retrieval performance using Recall@K.

    Args:
        retrieval_results (dict): Dictionary containing query IDs and their retrieved document IDs.
        ground_truth (dict): Dictionary containing the correct document IDs for each query.
        k (int): The number of top documents to consider in the evaluation. Default is 2.

    Returns:
        float: The average Recall@K across all queries.

    Recall@K is a metric that measures the proportion of relevant documents found among the top-K 
    retrieved documents.
    It helps evaluate how well a system retrieves the correct documents within a certain cutoff.
    """
    recall_at_k = []
    for query, relevant_docs in ground_truth.items():
        retrieved_docs = retrieval_results.get(query, [])[:k]
        hits = len(set(relevant_docs).intersection(set(retrieved_docs)))
        recall = hits / len(relevant_docs) if relevant_docs else 0
        recall_at_k.append(recall)
        logger.info(f"Recall@K for %s: %f", query, recall)
    return sum(recall_at_k) / len(recall_at_k)


# Generation Evaluation: BLEU Score
generated_texts = {
    "query1": "Andrew Flintoff is the Head Coach of Lions Cricket Club.",
    "query2": "Supreme Court urged Doctors to start the work.",
}

reference_texts = {
    "query1": "Flintoff named England Lions head coach.",
    "query2": "Supreme Court slams Kolkata police for delay in registering FIR; doctors are told to join work.",
}


def evaluate_generation(generated_texts, reference_texts) -> float:
    """
    Evaluates the quality of generated text using the BLEU score.

    Args:
        generated_texts (dict): Dictionary containing query IDs and the generated responses.
        reference_texts (dict): Dictionary containing query IDs and the reference (ground truth)
        responses.

    Returns:
        float: The average BLEU score across all queries.

    The BLEU score is a metric for evaluating text generation quality, comparing 
    generated text with a reference.
    A higher score indicates better alignment between generated text and the reference.
    """
    bleu_scores = []
    bleu_score = None
    smoothing = SmoothingFunction()
    for query, generated_text in generated_texts.items():
        reference = reference_texts.get(query, "").lower().split()
        generated = generated_text.lower().split()
        bleu_score = sentence_bleu(
            [reference], generated, smoothing_function=smoothing.method1
        )
        bleu_scores.append(bleu_score)
        logger.info("BLEU score for %s: %f", query, bleu_score)
    return sum(bleu_scores) / len(bleu_scores)


# Perform evaluations
recall_k = evaluate_retrieval(retrieval_results, ground_truth_retrieval, k=2)
bleu_score = evaluate_generation(generated_texts, reference_texts)

logger.info("Final Recall@K: %f", recall_k)
logger.info("Final BLEU Score: %f", bleu_score)
