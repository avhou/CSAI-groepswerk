from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.legacy.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.program import LMFormatEnforcerPydanticProgram
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from promt_reader import *
from typing import List
from tqdm import tqdm
from models import *
from huggingface_hub import login

import numpy as np
import pandas as pd

import json
import asyncio
import re
import os
from transformers import AutoTokenizer


# function that takes the report and creates the retriever (with indexes etc.)
def createRetriever(report, chunk_size, chunk_overlap, top_k):
    # load in document
    documents = SimpleDirectoryReader(input_files=[report]).load_data()
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # tries to keep sentences together
    nodes = parser.get_nodes_from_documents(documents)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    # build indexes
    print(f"creating embedding model")
    # embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
    login(os.getenv("HUGGINGFACE_TOKEN"))
    embed_model = HuggingFaceEmbedding(model_name="mistralai/Mistral-7B-v0.1", tokenizer=tokenizer, device="cpu")
    print(f"creating vector store")
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        show_progress=True
    )

    # configure retriever
    print(f"creating retriever")
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )
    return retriever


def basicInformation(retriever, model):
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

    prompt = read_general_prompt(model, sources_block)

    # or easily convert to message prompts (for chat API)
    # messages = qa_template.format_messages(sources=sources_block)

    # get response
    response = Ollama(temperature=0, model=model, request_timeout=120.0).complete(prompt)
    # replace front or back ```json {} ```
    response_text_json = response.text.replace("```json", "").replace("```", "")
    print("this is the response text before selection:")
    print(response_text_json)
    response_text = json.loads(response_text_json)

    print("this is the response text:")
    print(response_text)

    # create a text to it
    basic_info = f" - Company name: {response_text['COMPANY_NAME']}\n - Industry: {response_text['COMPANY_SECTOR']}\n - Headquarter Location: {response_text['COMPANY_LOCATION']}"

    print(basic_info)
    return basic_info, response_text

def yearInformation(retriever, model):
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

    prompt = read_year_prompt(model, sources_block)
    # llm = Ollama(temperature=0, model=model, request_timeout=120.0)
    login(os.getenv("HUGGINGFACE_TOKEN"))
    llm = HuggingFaceLLM(model_name="mistralai/Mistral-7B-v0.1")
    program = LMFormatEnforcerPydanticProgram(
        output_cls=YearQueryResponse,
        prompt_template_str=prompt,
        llm=llm,
        verbose=True,
    )

    response = program()
    print(response)

    # # or easily convert to message prompts (for chat API)
    # # messages = qa_template.format_messages(sources=sources_block)
    #
    # # get response
    # response = Ollama(temperature=0, model=model, request_timeout=120.0).complete(prompt)
    # # replace front or back ```json {} ```
    # response_text_json = response.text.replace("```json", "").replace("```", "")
    response_text = json.loads(response)

    print(f"this is the response text for year.txt information")
    print(response_text)

    return response_text


def createPromptTemplate(retriever, model, basic_info, query_str, explanation, answer_length):
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

    prompt = read_qa_prompt(model, basic_info, sources_block, query_str, explanation, answer_length)

    return prompt


def createPrompts(retriever, model, basic_info, answer_length, masterfile):
    prompts = []
    questions = []
    for i in tqdm(np.arange(0, masterfile.shape[0])):
        QUERY_STR = masterfile.iloc[i]["question"]
        questions.append(QUERY_STR)
        EXPLANATION = masterfile.iloc[i]["question definitions"]
        prompts.append(
            createPromptTemplate(retriever, model, basic_info, QUERY_STR, EXPLANATION, answer_length))
    print("Prompts Created")
    return prompts, questions

def createAnswers(prompts, model) -> List[str]:
    answers = []
    llm = Ollama(temperature=0, model=model, request_timeout=120.0)
    for p in tqdm(prompts):
        response = llm.complete(p)
        print("this is the main response:")
        print(response)
        answers.append(response)

    print("Answers Given")
    return answers


def outputExcel(answers, questions, prompts, report, masterfile, model, option="", excels_path="Excels_SustReps"):
    # create the columns
    categories, ans, ans_verdicts, source_pages, source_texts = [], [], [], [], []
    subcategories = [i.split("_")[1] for i in masterfile.identifier.to_list()]
    for i, a in enumerate(answers):
        try:
            # replace front or back ```json {} ```
            a = a.text.replace("```json", "").replace("```", "")
            answer_dict = json.loads(a)
        except:
            print(f"{i} with formatting error")
            try:
                answer_dict = {"ANSWER": "CAUTION: Formatting error occurred, this is the raw answer:\n" + a.text,
                               "SOURCES": "See In Answer"}
            except:
                answer_dict = {"ANSWER": "Failure in answering this question.", "SOURCES": "NA"}

        # final verdict
        verdict = re.search(r"\[\[([^]]+)\]\]", answer_dict["ANSWER"])
        if verdict:
            ans_verdicts.append(verdict.group(1))
        else:
            ans_verdicts.append("NA")

        # other values
        ans.append(answer_dict["ANSWER"])
        source_pages.append(", ".join(map(str, answer_dict["SOURCES"])))
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
    BASIC_INFO, response_text = basicInformation(retriever, model)
    print(response_text)
    print(BASIC_INFO)
    year_info = yearInformation(retriever, model)
    response_text["YEAR"] = year_info["YEAR"]
    response_text["REPORT_NAME"] = report
    print(response_text)

    prompts, questions = createPrompts(retriever, model, BASIC_INFO, answer_length, masterfile)

    answers = createAnswers(prompts, model)

    excels_path = f"Excel_Output_{model}"
    if not os.path.exists(excels_path):
        print("create excel output path")
        os.makedirs(excels_path)
    option = f"_topk{top_k}_params{less}"
    path_excel = outputExcel(answers, questions, prompts, report, masterfile, model, option, excels_path)

asyncio.run(main())
