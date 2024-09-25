import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from langchain_core.output_parsers import JsonOutputParser
from compiler.IR.llm import Demonstration
from llm.parsers import ColumnSelectionOutput

system_prompt = \
"""You are an expert and very smart data analyst.
Your task is to examine the provided database schema, understand the posed question, and use the hint to pinpoint the specific columns within tables that are essential for crafting a SQL query to answer the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the columns that will be used in the SQL query.

Task:
Based on the database schema, question, and hint provided, your task is to identify all and only the columns that are essential for crafting a SQL query to answer the question.
For each of the selected columns, explain why exactly it is necessary for answering the question. Your reasoning should be concise and clear, demonstrating a logical connection between the columns and the question asked.

Tip: If you are choosing a column for filtering a value within that column, make sure that column has the value as an example."""

inputs = ["QUESTION", "HINT"]

output_format = 'response'

output_format_instructions = \
"""Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Your reasoning for selecting the columns, be concise and clear.",
  "table_name1": ["column1", "column2", ...],
  "table_name2": ["column1", "column2", ...],
  ...
}}
```

Make sure your response includes the table names as keys, each associated with a list of column names that are necessary for writing a SQL query to answer the question.
For each aspect of the question, provide a clear and concise explanation of your reasoning behind selecting the columns.
Take a deep breath and think logically. If you do the task correctly, I will give you 1 million dollars.

Only output a json as your response."""

demos = []

registry = {}

def get_executor(schema_string: str):
    if schema_string in registry:
        return registry[schema_string]
    semantic = LangChainSemantic(
        system_prompt=system_prompt.format(DATABASE_SCHEMA=schema_string),
        inputs=inputs,
        output_format=output_format,
        output_format_instructions=output_format_instructions,
        demos=demos
    )
    exec = LangChainLM('column_selection', semantic, opt_register=True)
    exec.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}
    runnable_exec = exec.as_runnable()
    registry[schema_string] = runnable_exec
    return runnable_exec | JsonOutputParser()