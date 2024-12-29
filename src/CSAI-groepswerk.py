from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from promt_reader import *
from typing import List
from tqdm import tqdm
from models import *

import numpy as np
import pandas as pd

import json
import asyncio
import re
import os


# function that takes the report and creates the retriever (with indexes etc.)
def createRetriever(report: str, chunk_size: int, chunk_overlap: int, top_k: int) -> VectorIndexRetriever:
    # load in document
    documents = SimpleDirectoryReader(input_files=[report]).load_data()
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # tries to keep sentences together
    nodes = parser.get_nodes_from_documents(documents)


    # build indexes
    print(f"creating embedding model")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    vector_store_path = os.path.abspath("../vector-store")

    print(f"creating vector store")
    if os.path.exists(vector_store_path):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=vector_store_path), embed_model=embed_model)
    else:
        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=vector_store_path)

    # configure retriever
    print(f"creating retriever")
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    return retriever


def basicInformation(retriever: VectorIndexRetriever, model: str):
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

    print(f"sources block has length {len(sources)}")

    json_schema = str(GeneralInfoQueryResponse.model_json_schema())
    print(f"general info schema : {json_schema}")

    prompt = read_general_prompt(model, sources_block, json_schema)

    response = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True).complete(prompt)

    response_json = json.loads(response.text)
    print(f"basic info response : {response_json}")
    # create textual representation
    basic_info = f" - Company name: {response_json['COMPANY_NAME']}\n - Industry: {response_json['COMPANY_SECTOR']}\n - Headquarter Location: {response_json['COMPANY_LOCATION']}"
    print(f"basic info text : {basic_info}")

    return basic_info, response_json

def yearInformation(retriever: VectorIndexRetriever, model: str):
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
    print(f"year schema : {json_schema}")

    prompt = read_year_prompt(model, sources_block, json_schema)

    response = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True).complete(prompt)
    response_json = json.loads(response.text)
    print(f"year response : {response_json}")
    return response_json


def createPromptTemplate(retriever: VectorIndexRetriever, model: str, basic_info: str, query_str: str, explanation: str, answer_length: int) -> str:
    # Query content
    retrieved_nodes = retriever.retrieve(query_str)
    # create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata['page_label']
        # remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    json_schema = str(QueryResponse.model_json_schema())
    print(f"qa schema : {json_schema}")

    prompt = read_qa_prompt(model, basic_info, sources_block, query_str, explanation, answer_length, json_schema)

    return prompt


def createPrompts(retriever: VectorIndexRetriever, model: str, basic_info: str, answer_length: int, masterfile):
    prompts = []
    questions = []
    for i in np.arange(0, masterfile.shape[0]):
        query_str = masterfile.iloc[i]["question"]
        questions.append(query_str)
        explanation = masterfile.iloc[i]["question definitions"]
        prompts.append(
            createPromptTemplate(retriever, model, basic_info, query_str, explanation, answer_length))
    print("Prompts Created")
    return prompts, questions

def createAnswers(prompts: List[str], model: str) -> List[str]:
    answers = []
    llm = Ollama(temperature=0, model=model, request_timeout=120.0, json_mode=True)
    for p in tqdm(prompts):
        response = llm.complete(p)
        print(f"qa response : {response.text}")
        answers.append(response)

    print("Answers Given")
    return answers


def outputExcel(answers, questions, prompts, report, masterfile, model, option="", excels_path="Excels_SustReps"):
    # create the columns
    categories, ans, ans_verdicts, source_pages, source_texts = [], [], [], [], []
    subcategories = [i.split("_")[1] for i in masterfile.identifier.to_list()]
    for i, a in enumerate(answers):
        format_error = False
        try:
            # replace front or back ```json {} ```
            a = a.text.replace("```json", "").replace("```", "")
            print(f"checking response {a}")
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
    df_out = pd.DataFrame(
        {"category": categories, "subcategory": subcategories, "question": questions, "decision": ans_verdicts,
         "answer": ans,
         "source_pages": source_pages, "source_texts": source_texts})
    excel_path_qa = f"./{excels_path}/" + report.split("/")[-1].split(".")[0] + f"_{model}" + f"{option}" + ".xlsx"
    df_out.to_excel(excel_path_qa)
    return excel_path_qa

async def main():
    masterfile = pd.read_excel("../data/questions_masterfile_100524.xlsx")
    chunk_size = 350
    chunk_overlap = 50
    top_k = 8
    answer_length = 200

    report = "../data/CSAI-reports/Zara_Financial_Sustainability_Report_2023.pdf"
    model = "mistral"

    # if option of less is given
    try:
        less = int(os.getenv("NUMBER_OF_QUESTIONS"))
        masterfile = masterfile[:less].copy()
        print(f"Execution with subset of {less} indicators.")
    except:
        less = "all"
        print("Execution with all parameters.")

    retriever = createRetriever(report, chunk_size, chunk_overlap, top_k)
    basic_info, response_text = basicInformation(retriever, model)
    year_info = yearInformation(retriever, model)
    response_text["YEAR"] = year_info["YEAR"]
    response_text["REPORT_NAME"] = report
    print(response_text)

    prompts, questions = createPrompts(retriever, model, basic_info, answer_length, masterfile)

    answers = createAnswers(prompts, model)

    excels_path = f"Excel_Output_{model}"
    if not os.path.exists(excels_path):
        print("create excel output path")
        os.makedirs(excels_path)
    option = f"_topk{top_k}_params{less}"
    path_excel = outputExcel(answers, questions, prompts, report, masterfile, model, option, excels_path)

asyncio.run(main())
