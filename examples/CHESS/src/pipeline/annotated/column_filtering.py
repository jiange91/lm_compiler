import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import Demonstration
from llm.parsers import ColumnFilteringOutput
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

system_prompt = \
"""You are a Careful data scientist.
In the following, you will be given a set of information about a column in a database, a question asked about the database, and a hint regarding the question.

Your task is to determine whether the column information is relevant to the question and the hint.
To achieve the task, you need to follow the following steps:
- First, thoroughly review the information provided for the column. 
- Next, understand the database question and the hint associated with it. 
- Based on your analysis, determine whether the column information is relevant to the question and the hint.

Make sure to keep the following points in mind:
- You are only given one column information, which is not enough to answer the question. So don't worry about the missing information and only focus on the given column information.
- If you see a keyword in the question or the hint that is present in the column information, consider the column as relevant.
- Pay close attention to the "Example of values in the column", and if you see a connection between the values and the question, consider the column as relevant."""

inputs = ["COLUMN_PROFILE", "QUESTION", "HINT"]

output_format = ColumnFilteringOutput


semantic = LangChainSemantic(
    system_prompt=system_prompt,
    inputs=inputs,
    output_format="is_column_information_relevant",
    output_format_instructions="Please response with Yes or No (no other text should be included).",
)
exec = LangChainLM('column_filtering', semantic, opt_register=True)
raw_runnable_exec = exec.as_runnable() | StrOutputParser()

@chain
def runnable_exec(inputs):
    is_relevant = raw_runnable_exec.invoke(inputs)
    return {
      "chain_of_thought_reasoning": "",
      "is_column_information_relevant": is_relevant,
    }
