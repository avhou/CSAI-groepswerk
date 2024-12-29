import os
from llama_index.core.prompts.base import PromptTemplate

def read_general_prompt(model: str, sources: str, json_schema: str) -> str:
    template = PromptTemplate(read_prompt(model, "general.txt"))
    return template.format(sources=sources, json_schema=json_schema)

def read_year_prompt(model: str, sources: str, json_schema: str) -> str:
    template = PromptTemplate(read_prompt(model, "year.txt"))
    return template.format(sources=sources, json_schema=json_schema)

def read_qa_prompt(model: str, basic_info: str, sources: str, question: str, explanation: str, answer_length: int, json_schema: str) -> str:
    template = PromptTemplate(read_prompt(model, "qa.txt"))
    return template.format(basic_info=basic_info, sources=sources, question=question, explanation=explanation, answer_length=answer_length, json_schema=json_schema)


def read_prompt(model: str, file: str) -> str:
    with open(os.path.join("prompts", model.split(":")[0], file), "r") as f:
        return f.read()
