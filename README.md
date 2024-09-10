# RAG Evaluation Task

# RAG Model
RAG is a hybrid model that combines two components:

1. **Retriever**: Fetches relevant documents from a knowledge base (like Wikipedia or a custom database) based on a given query.
2. **Generator**: Uses the retrieved documents to generate a response or text based on the input query.

## Evaluation of RAG Systems
Evaluating a RAG model involves assessing both the retrieval quality and the generation quality. Different evaluation metrics are used for each of these phases.

## Types of RAG Evaluation
1. **Retrieval Evaluation**:
   - **Recall@K**: Measures how often the relevant document is present in the top K retrieved documents. This is crucial for understanding if the retriever is bringing back relevant content. High Recall@K implies the retriever finds useful documents for the generator.
   - **Precision@K**: Measures how many of the top K documents are actually relevant. While Recall@K focuses on relevance, Precision@K focuses on the proportion of the retrieved documents that are accurate.
   - **MRR (Mean Reciprocal Rank)**: MRR (Mean Reciprocal Rank): Measures the position of the first relevant document among the retrieved results. If the relevant document is retrieved at a high rank, MRR will be higher.

2. **Generation Evaluation**:
   - **BLEU**: Evaluates the quality of the generated text by comparing it to a reference (ground-truth) text. BLEU focuses on n-gram overlap between the generated text and the reference text.
   - **ROUGE**: Measures how much of the reference text is captured by the generated text. It is often used for summarization tasks.
   - **Perplexity**: Measures how uncertain the model is when predicting the next word. Lower perplexity means the model is more confident in its predictions.

## Chosen Metrics
1. **Recall@K**: 
    - Evaluates how well the retriever returns relevant documents. I chose this because retrieving accurate documents is essential for the generator to produce useful outputs.
    - Recall@K tells us whether the retriever component is bringing back the correct and relevant documents from the knowledge base.
    - This is important because the generation phase relies heavily on having the right documents to base its response on. If the retriever fetches irrelevant documents, the quality of the generated response will suffer.
    - K=2: In this case, I evaluate whether the correct document is retrieved within the top 2 results.
2. **BLEU Score**: 
    - Evaluates how close the generated text is to a reference text. This measures the quality and relevance of the generation.
    - BLEU is commonly used in natural language generation tasks to evaluate the quality of generated responses.
    - By comparing the generated text to a reference (true) text, we can measure how close the generated text is to the desired output.
    - This metric ensures that the generator is producing text that is both grammatically correct and relevant to the retrieved information.

## Instructions
To run the program:
1. Install the required libraries in virtual environment:
2. `python evaluate.py`

This program evaluates a basic Retrieval-Augmented Generation (RAG) system using two metrics: **Recall@K** and **BLEU**.
1. **evaluate_retrieval**: This function calculates Recall@K for each query. It logs the individual and overall scores.
2. **evaluate_generation**: This function computes the BLEU score for the generated text. It logs individual and overall scores.
3. **Logging**: The evaluation results are logged at each stage.