import os
from typing import List, Optional

from llama_index.llms.huggingface import HuggingFaceLLM
import csv

def read_queries_as_list(file_path: str) -> List[str]:
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        return [row for row in reader]

HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
MODEL_NAME: Optional[str] = os.getenv("MODEL_NAME")
csv_file_path: Optional[str] = os.getenv("CSV_FILE_PATH")

locally_run = HuggingFaceLLM(model_name=MODEL_NAME, hf_token=HF_TOKEN)

queries = read_queries_as_list(csv_file_path)
