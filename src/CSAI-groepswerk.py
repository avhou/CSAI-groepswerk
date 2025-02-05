from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import CompletionResponse
from llama_index.core.evaluation import FaithfulnessEvaluator, ContextRelevancyEvaluator, CorrectnessEvaluator, SemanticSimilarityEvaluator, GuidelineEvaluator, EvaluationResult
from llama_index.core.schema import NodeWithScore
from promt_reader import *
import logging
import sys
import csv

from tqdm import tqdm
from models import *
from typing import Collection, Dict, Optional

import numpy as np
import pandas as pd

import json
import asyncio
import os

def get_embedding_model(model_name:str="nomic-embed-text", base_url:str="http://localhost:11434"):
    return OllamaEmbedding(model_name, base_url)

# function that takes the report and creates the retriever (with indexes etc.)
def create_retriever(report: Collection[str], chunk_size: int, chunk_overlap: int, top_k: int) -> VectorIndexRetriever:
    # load in document
    documents = SimpleDirectoryReader(input_files=report).load_data()
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # tries to keep sentences together
    nodes = parser.get_nodes_from_documents(documents)

    # build indexes
    embed_model = get_embedding_model()
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        show_progress=True,
    )

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    return retriever


def basic_information(retriever: VectorIndexRetriever, model: str):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "What is the name of the company, the sector it operates in and location of headquarters?")
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    json_schema = str(GeneralInfoQueryResponse.model_json_schema())
    prompt = read_general_prompt(model, sources_block, json_schema)

    response = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True, duration=-1, context_size=4096).complete(prompt)

    response_json = json.loads(response.text)
    # create textual representation
    basic_info = f" - Company name: {response_json['COMPANY_NAME']}\n - Industry: {response_json['COMPANY_SECTOR']}\n - Headquarter Location: {response_json['COMPANY_LOCATION']}"
    return basic_info, response_json

def year_information(retriever: VectorIndexRetriever, model: str):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "In which year.txt was the report published?")
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    json_schema = str(YearQueryResponse.model_json_schema())
    prompt = read_year_prompt(model, sources_block, json_schema)

    response = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True, duration=-1, context_size=4096).complete(prompt)
    response_json = json.loads(response.text)
    return response_json

def create_prompt_template(retrieved_nodes: List[NodeWithScore], model: str, basic_info: str, query_str: str, explanation: str, answer_length: int) -> str:
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    json_schema = str(QueryResponse.model_json_schema())

    prompt = read_qa_prompt(model, basic_info, sources_block, query_str, explanation, answer_length, json_schema)

    return prompt


def create_prompts(retriever: VectorIndexRetriever, model: str, basic_info: str, answer_length: int, masterfile):
    prompts = []
    contexts = []
    questions = []
    for i in tqdm(np.arange(0, masterfile.shape[0]),desc="Retrieving prompts"):
        query_str = masterfile.iloc[i]["question"]
        questions.append(query_str)
        explanation = masterfile.iloc[i]["question definitions"]
        retrieved_nodes = retriever.retrieve(query_str)
        contexts.append([node.get_content().replace("\n", "") for node in retrieved_nodes])
        prompts.append(
            create_prompt_template(retrieved_nodes, model, basic_info, query_str, explanation, answer_length))
    return prompts, questions, contexts

def create_answers(prompts: Collection[str], model: str) -> Collection[CompletionResponse]:
    answers = []
    llm = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True, duration=-1, context_size=4096)
    for p in tqdm(prompts, desc="Generating responses"):
        response = llm.complete(p)
        answers.append(response)

    print(f"number of answers: {len(answers)}")
    return answers


def output_excel(answers, questions, prompts, report, masterfile, model, evaluation_metrics: Dict[str, Collection[EvaluationResult]], option="", excels_path="Excels_SustReps"):
    # create the columns
    categories, ans, ans_verdicts, source_pages, source_texts = [], [], [], [], []
    subcategories = [i.split("_")[1] for i in masterfile.identifier.to_list()]
    for i, a in enumerate(answers):
        format_error = False
        try:
            # replace front or back ```json {} ```
            a = a.text.replace("```json", "").replace("```", "")

            answer_dict = json.loads(a)
            # check for right format
            QueryResponse.model_validate_json(a)
        except Exception as e:
            print(f"{i} with formatting error : {e}")
            format_error = True
            try:
                answer_dict = {"ANSWER": "CAUTION: Formatting error occurred, this is the raw answer:\n" + a.text,
                               "EXPLANATION": f"See In Answer.  Error was {e}.",
                               "SOURCES": "See In Answer"}
            except:
                answer_dict = {"ANSWER": "Failure in answering this question.", "EXPLANATION": "See In Answer", "SOURCES": []}

        if format_error:
            ans_verdicts.append("N/A")
        else:
            if answer_dict["ANSWER"]:
                ans_verdicts.append("YES")
            else:
                ans_verdicts.append("NO")

        # other values
        ans.append(answer_dict["EXPLANATION"])
        source_pages.append(", ".join(map(str, answer_dict["SOURCES"] if "SOURCES" in answer_dict else [])))
        source_texts.append(prompts[i].split("---------------------")[1])

        if i == 0:
            category = "target"
        if i == 12:
            category = "governance"
        if i == 21:
            category = "strategy"
        if i == 45:
            category = "tracking"
        categories.append(category)

    # create DataFrame and export as excel
    df_out = pd.DataFrame({
        "category": categories,
        "subcategory": subcategories,
        "question": questions,
        "decision": ans_verdicts,
        "answer": ans,
        "source_pages": source_pages,
        "source_texts": source_texts,
        "faithfulness_score": [result.score for result in evaluation_metrics["Faithfulness"]],
        "faithfulness_feedback": [result.feedback for result in evaluation_metrics["Faithfulness"]],
        "context_relevancy_score": [result.score for result in evaluation_metrics["Context_relevancy"]],
        "context_relevancy_feedback": [result.feedback for result in evaluation_metrics["Context_relevancy"]],
        "correctness_score": [result.score for result in evaluation_metrics["Correctness"]],
        "correctness_feedback": [result.feedback for result in evaluation_metrics["Correctness"]],
        "semantic_semilarity_score": [result.score for result in evaluation_metrics["Semantic_semilarity"]],
        "semantic_semilarity_feedback": [result.feedback for result in evaluation_metrics["Semantic_semilarity"]]
        })
    excel_path_qa = f"./{excels_path}/" + report.split("/")[-1].split(".")[0] + f"_{model}" + f"{option}" + ".xlsx"
    df_out.to_excel(excel_path_qa)
    return excel_path_qa

async def evaluate_model(evaluating_llm_name: str, queries: Collection[str], references_per_query: Collection[str], answers: Collection[str], ground_truth_labels: Collection[str]):
    metrics = {}
    metrics["Faithfulness"] = await evaluate_faithfulness(evaluating_llm_name,queries, references_per_query, answers)
    metrics["Context_relevancy"] = await evaluate_context_relevancy(evaluating_llm_name,queries, references_per_query, answers)
    metrics["Correctness"]  = await evaluate_correctness(evaluating_llm_name, queries, references_per_query, answers, ground_truth_labels)
    metrics["Semantic_semilarity"] = await evaluate_semantic_semilarity(queries, references_per_query, answers, ground_truth_labels)
    return metrics

async def evaluate_correctness(evaluating_llm_name: str, queries: Collection[str], references_per_query: Collection[str], answers: Collection[CompletionResponse], ground_truth_labels: Collection[str]):
    results = []
    evaluating_llm = Ollama(temperature=0, model=evaluating_llm_name, request_timeout=120.0, duration=-1, context_size=4096)

    evaluator = CorrectnessEvaluator(evaluating_llm)
    print(f"number of labels: {len(ground_truth_labels)}")
    for index, _ in tqdm(enumerate(answers), desc="Evaluating correctness"):
        query, answer, references, ground_truth = queries[index], answers[index], references_per_query[index], ground_truth_labels[index][0]
        evaluation_result = await evaluator.aevaluate(query=query, response=answer.text, contexts=references, reference=ground_truth)
        results.append(evaluation_result)

    return results

async def evaluate_semantic_semilarity(queries: Collection[str], references_per_query: Collection[str], answers: Collection[CompletionResponse], ground_truth_labels: Collection[str]):
    results = []
    embedding_model = get_embedding_model()

    evaluator = SemanticSimilarityEvaluator(embed_model=embedding_model)

    for index, _ in tqdm(enumerate(answers), desc="Evaluating semantic semilarity"):
        query, answer, references, ground_truth = queries[index], answers[index], references_per_query[index], ground_truth_labels[index][0]
        evaluation_result = await evaluator.aevaluate(query=query, response=answer.text, contexts=references, reference=ground_truth)
        results.append(evaluation_result)

    return results


async def evaluate_faithfulness(evaluating_llm_name: str, queries: Collection[str], references_per_query: Collection[str], answers: Collection[CompletionResponse]):
    results = []
    evaluating_llm = Ollama(temperature=0, model=evaluating_llm_name, request_timeout=120.0, duration=-1, context_size=4096)

    evaluator = FaithfulnessEvaluator(llm=evaluating_llm)
    
    for index, _ in tqdm(enumerate(answers), desc="Evaluating faithfullness"):
        query, answer, references = queries[index], answers[index], references_per_query[index]
        evaluation_result = await evaluator.aevaluate(query=query, response=answer.text, contexts=references)
        results.append(evaluation_result)

    return results 

async def evaluate_context_relevancy(evaluating_llm_name: str, queries: Collection[str], references_per_query: Collection[str], answers: Collection[CompletionResponse]):
    results = []
    evaluating_llm = Ollama(temperature=0, model=evaluating_llm_name, request_timeout=120.0, duration=-1, context_size=4096)

    evaluator = ContextRelevancyEvaluator(llm=evaluating_llm)
    
    for index, _ in tqdm(enumerate(answers), desc="Evaluating context_relevancy"):
        query, answer, references = queries[index], answers[index], references_per_query[index]
        evaluation_result = await evaluator.aevaluate(query=query, response=answer.text, contexts=references)
        results.append(evaluation_result)

    return results 

async def get_ground_truth(prompts: Collection[str], ground_truth_label_path: str = "../data/ground_truth.csv") -> List[str]:
    # If CSV exists, read and return its contents
    if os.path.exists(ground_truth_label_path):
        with open(ground_truth_label_path, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            all_labels = [row for row in csv_reader]
            return all_labels

    # If no ground-truth labels found, generate them
    print("No ground truth labels found. Switching to generating ground truth.")
    ground_truth_responses = create_answers(prompts, "llama3:instruct")

    # Clean up the text and parse JSON
    ground_truth_strings = [
        response.text.replace("```json", "").replace("```", "")
        for response in ground_truth_responses
    ]
    ground_truth_dicts = [json.loads(string) for string in ground_truth_strings]

    # Extract labels
    labels = [d["EXPLANATION"] for d in ground_truth_dicts]

    # Write the labels to CSV
    with open(ground_truth_label_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for label in labels:
            # each label is a string, so wrap it in a list
            writer.writerow([label])
        
    return labels

async def main():
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    masterfile = pd.read_excel("../data/questions_masterfile_100524.xlsx")
    chunk_size = 350
    chunk_overlap = 50
    top_k = 8
    answer_length = 200
    evaluating_llm_name = "llama3:instruct"

    report_sets = [
        # ["../data/CSAI-reports/Zara_Financial_Sustainability_Report_2023.pdf"],
        [
            "../data/CSAI-reports/HM-Group-Annual-and-Sustainability-Report-2023.pdf",
            "../data/CSAI-reports/HM-Group-Sustainability-Disclosure-2023.pdf",
        ]
    ]
    excel_sets = [
       # "zara",
        "hm",
    ]
    models = [
        "mistral",
        "llama2",
        "llama3:instruct",
        "phi3:14b-instruct",
    ]

    for i, report_set in enumerate(report_sets):
        for model in models:
            # if option of less is given
            try:
                less = int(os.getenv("NUMBER_OF_QUESTIONS"))
                masterfile = masterfile[:less].copy()
                print(f"Execution with subset of {less} indicators.")
            except:
                less = "all"
                print("Execution with all parameters.")

            retriever = create_retriever(report_set, chunk_size, chunk_overlap, top_k)
            basic_info, response_text = basic_information(retriever, model)
            year_info = year_information(retriever, model)
            response_text["YEAR"] = year_info["YEAR"]
            response_text["REPORT_NAME"] = excel_sets[i]

            prompts, queries, references_per_query = create_prompts(retriever, model, basic_info, answer_length, masterfile)

            answers = create_answers(prompts, model)

            ground_truth = await get_ground_truth(prompts)
            evaluation_metrics = await evaluate_model(evaluating_llm_name, queries, references_per_query, answers, ground_truth)

            excels_path = f"Excel_Output_{model.split(':')[0]}"
            if not os.path.exists(excels_path):
                print("create excel output path")
                os.makedirs(excels_path)
            option = f"_topk{top_k}_params{less}"
            path_excel = output_excel(answers, queries, prompts, excel_sets[i], masterfile, model, evaluation_metrics, option, excels_path)
            print(f"excel was written to {path_excel}")

asyncio.run(main())
