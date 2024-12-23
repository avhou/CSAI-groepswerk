import os
from llama_index.core.prompts.base import PromptTemplate

def read_general_prompt(model: str, sources: str) -> str:
    template = PromptTemplate(read_prompt(model, "general.txt"))
    return template.format(sources=sources)

def read_year_prompt(model: str, sources: str) -> str:
    template = PromptTemplate(read_prompt(model, "year.txt"))
    return template.format(sources=sources)

def read_qa_prompt(model: str, basic_info: str, sources: str, question: str, explanation: str, answer_length: str) -> str:
    template = PromptTemplate(read_prompt(model, "qa.txt"))
    return template.format(basic_info=basic_info, sources=sources, question=question, explanation=explanation, answer_length=answer_length)


def read_prompt(model: str, file: str) -> str:
    with open(os.path.join("prompts", model, file), "r") as f:
        return f.read()
