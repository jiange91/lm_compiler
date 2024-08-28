from typing import Literal
import concurrent.futures

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.modules import Map, CodeBox

#------------------------ Retrieval Grader ------------------------#
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
# ------------------------ Old kernel ------------------------#
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


# Post-processing
def format_docs(docs):
    return "\n\n".join(f'Document {i+1}:\n' + doc for i, doc in enumerate(docs))

def process_single_question(question, retrieval_grader, docs):
    inputs = [{"question": question, "document": d} for d in docs]
    scores = retrieval_grader.batch(inputs)
    filtered_docs = []
    for doc, score in zip(docs, scores):
        if score.binary_score == "yes":
            filtered_docs.append(doc)
    return format_docs(filtered_docs)

def doc_filter_kernel(llm, sub_questions, raw_docs):
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader

    documents_str = []

    with concurrent.futures.ThreadPoolExecutor() as outer_executor:
        outer_futures = [
            outer_executor.submit(
                process_single_question,
                question,
                retrieval_grader,
                docs
            )
            for question, docs in zip(sub_questions, raw_docs)
        ]

        for future in outer_futures:
            try:
                documents_str.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    return {'documents': documents_str}

# ------------------------ New Semantic ------------------------#
def sub_question_map_kernel(sub_questions, raw_docs):
    for sub_question, docs in zip(sub_questions, raw_docs):
        yield {"sub_question": sub_question, "docs_for_filter": docs}
        
def doc_map_kernel(sub_question, docs_for_filter):
    for doc in docs_for_filter:
        yield {"sub_question": sub_question, "doc_for_filter": doc}

doc_filter_semantic = LangChainSemantic(
    system_prompt=system,
    inputs=["sub_question", "doc_for_filter"],
    output_format=GradeDocuments,
)

doc_filter_lm = LangChainLM(
    name="doc_filter_lm",
    semantic=doc_filter_semantic,
)

doc_mapper = Map(
    name="doc_mapper",
    sub_graph=doc_filter_lm,
    map_kernel=doc_map_kernel,
    output_fields="binary_score",
    max_parallel=5,
)

sub_question_mapper = Map(
    name="sub_question_mapper",
    sub_graph=doc_mapper,
    map_kernel=sub_question_map_kernel,
    output_fields="binary_score",
    max_parallel=5,
)

def post_process_docs(binary_score, raw_docs):
    documents_str = []
    for docs, scores in zip(raw_docs, binary_score):
        filtered_docs = []
        for doc, score in zip(docs, scores):
            if score == "yes":
                filtered_docs.append(doc)
        documents_str.append(format_docs(filtered_docs))
    return {'documents': documents_str}

doc_filter_post_process = CodeBox(
    name="doc_filter_post_process",
    kernel=post_process_docs,
)