from typing import Literal
import concurrent.futures

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

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
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def process_single_question(question, retrieval_grader, docs):
    inputs = [{"question": question, "document": d.page_content} for d in docs]
    scores = retrieval_grader.batch(inputs)
    filtered_docs = []
    for doc, score in zip(docs, scores):
        if score.binary_score == "yes":
            filtered_docs.append(doc)
    return format_docs(filtered_docs)

def doc_filter_kernel(sub_questions, raw_docs):
    llm = doc_filter_kernel.lm
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