from typing import Literal
# from vdb import retriever
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

query_format = """
{question}
{passage}
"""

retriever = None

def process_single_document(question, doc, retrieval_grader):
    score = retrieval_grader.invoke({"question": question, "document": doc.page_content}).binary_score
    if score == "yes":
        return doc
    return None

def process_single_question(question, passage, retrieval_grader, query_format):
    docs = retriever.get_relevant_documents(query_format.format(question=question, passage=passage))
    
    with concurrent.futures.ThreadPoolExecutor() as inner_executor:
        inner_futures = [
            inner_executor.submit(process_single_document, question, d, retrieval_grader)
            for d in docs
        ]
        
        filtered_docs = []
        for future in concurrent.futures.as_completed(inner_futures):
            result = future.result()
            if result is not None:
                filtered_docs.append(result)
                
    return format_docs(filtered_docs)

def single_thread_retrieve_filter_kernel(sub_questions, passages):
    llm = single_thread_retrieve_filter_kernel.lm
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader
    documents_str = []
    for question, passage in zip(sub_questions, passages):
        docs = retriever.get_relevant_documents(query_format.format(question=question, passage=passage))
        filtered_docs = []
        for d in docs:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score
            if score == "yes":
                filtered_docs.append(d)
            else:
                continue
        documents_str.append(format_docs(filtered_docs))
    return {'documents': documents_str}

def retrieve_filter_kernel(sub_questions, passages):
    llm = retrieve_filter_kernel.lm
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader

    documents_str = []

    with concurrent.futures.ThreadPoolExecutor() as outer_executor:
        outer_futures = [
            outer_executor.submit(
                process_single_question,
                question,
                passage,
                retrieval_grader,
                query_format
            )
            for question, passage in zip(sub_questions, passages)
        ]

        for future in concurrent.futures.as_completed(outer_futures):
            try:
                documents_str.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    return {'documents': documents_str}