import os
from dotenv import load_dotenv
from typing import List
from llama_index.llms.huggingface import HuggingFaceLLM
import csv
import torch

def read_queries_as_list(file_path: str) -> List[str]:
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        return [row[0] for row in reader] 

load_dotenv()

CSV_FILE_PATH = os.getenv('CSV_FILE_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')

print(f"CSV File Path: {CSV_FILE_PATH}, Model Name: {MODEL_NAME}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

locally_run = HuggingFaceLLM(model_name=MODEL_NAME, device_map=device)

queries = read_queries_as_list(CSV_FILE_PATH)

for query in queries:
    print(f"Query: {query}")

responses = locally_run.complete(queries)