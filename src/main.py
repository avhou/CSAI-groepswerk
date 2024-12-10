import os
from dotenv import load_dotenv
from typing import List
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.evaluation import FaithfulnessEvaluator, ContextRelevancyEvaluator, CorrectnessEvaluator, GuidelineEvaluator, SemanticSimilarityEvaluator
import csv
import torch

def read_queries_as_list(file_path: str) -> List[str]:
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader] 

load_dotenv()

CSV_FILE_PATH = os.getenv('CSV_FILE_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')

# Make sure your cuda version is equal to the cuda pytorch version
print(f"Pytorch version: {torch.version.cuda}")

print("Is CUDA available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

locally_run = HuggingFaceLLM(model_name=MODEL_NAME, device_map=device)

queries = read_queries_as_list(CSV_FILE_PATH)

for query in queries:
    print(f"Query: {query}")

responses = locally_run.complete(queries)

faithfullness_evaluator = FaithfulnessEvaluator()
context_relevancy_evaluator = ContextRelevancyEvaluator()
correctness_evaluator = CorrectnessEvaluator()
guideline_evaluator = GuidelineEvaluator()
semantic_similarity_evaluator = SemanticSimilarityEvaluator()

faithfullness_score = faithfullness_evaluator.evaluate(queries, responses)
context_relevancy_score = context_relevancy_evaluator.evaluate(queries, responses)
correctness_score = correctness_evaluator.evaluate(queries, responses)
guideline_score = guideline_evaluator.evaluate(queries, responses)
semantic_similarity_score = semantic_similarity_evaluator.evaluate(queries, responses)

print(f"Faithfullness Score: {faithfullness_score}")
print(f"Context Relevancy Score: {context_relevancy_score}")
print(f"Correctness Score: {correctness_score}")
print(f"Guideline Score: {guideline_score}")
print(f"Semantic Similarity Score: {semantic_similarity_score}")

