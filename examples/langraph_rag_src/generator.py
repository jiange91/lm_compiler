from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from concurrent.futures import ThreadPoolExecutor
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.modules import Map
from schemas import *

# ------------------------ Old Kernel ------------------------#
# Prompt
system = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = hub.pull("rlm/rag-prompt")


def generator_kernel(llm, sub_questions, documents):
    rag_chain = prompt | llm | StrOutputParser()
    inputs = [{"context": document, "question": question} for question, document in zip(sub_questions, documents)]
    sub_answers = rag_chain.batch(inputs)
    # for question, document in zip(sub_questions, documents):
    #     sub_answers.append(rag_chain.invoke({"context": document, "question": question}))
    return {'sub_answers': sub_answers}

# ------------------------ New Semantic ------------------------#
generator_system = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
"""


generator_semantic = LangChainSemantic(
    system_prompt=generator_system,
    inputs=['sub_question', 'context'],
    output_format=KnowledgeAnswer,
)

generator_lm = LangChainLM(
    name="sub_answer_generator",
    semantic=generator_semantic,
)

def sub_answer_map_kernel(sub_questions, documents):
    for question, document in zip(sub_questions, documents):
        yield {'sub_question': question, 'context': document}

sub_answer_mapper = Map(
    name="knowledge_curation",
    sub_graph=generator_lm,
    map_kernel=sub_answer_map_kernel,
    output_fields="knowledge",
)