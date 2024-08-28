from typing import Literal
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

subqs_format = NumberedListOutputParser()

#--------------------- decomposer ---------------------#
system = f"""
You are an expert at decomposing a user question into a list of sub-questions, each of which focus on a more dedicated aspect of the original question. If you think the question is already focused enough, just repeat the question as the only sub-question.
{subqs_format.get_format_instructions()}
"""

query_decompose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\nAnswer: \n"),
    ]
)


def decompose_kernel(question):
    llm = decompose_kernel.lm
    question_decompose = query_decompose_prompt | llm | subqs_format
    subqs = question_decompose.invoke({"question": question})
    return {'sub_questions': subqs}
    