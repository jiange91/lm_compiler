from typing import Literal
from vdb import retriever
import concurrent.futures   

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import logging

logger = logging.getLogger(__name__)
#------------------------ Retrieval ------------------------#

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

query_format = """
{question}
{passage}
"""


def retrieve_single_question(question, passage):
    docs = retriever.invoke(query_format.format(question=question, passage=passage))
    docs_str = [doc.page_content for doc in docs]
    return docs_str

def retrieve_kernel(sub_questions, passages):
    raw_docs = []
    with concurrent.futures.ThreadPoolExecutor() as outer_executor:
        outer_futures = [
            outer_executor.submit(
                retrieve_single_question,
                question,
                passage,
            )
            for question, passage in zip(sub_questions, passages)
        ]

        for future in outer_futures:
            try:
                raw_docs.append(future.result())
            except Exception as e:
                print(f"An error occurred: {e}")

    return {'raw_docs': raw_docs}