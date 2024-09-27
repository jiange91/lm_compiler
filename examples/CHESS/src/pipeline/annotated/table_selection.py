import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from langchain_core.output_parsers import JsonOutputParser
from compiler.IR.llm import Demonstration
from llm.parsers import TableSelectionOutputParser

system_prompt = \
"""You are an expert and very smart data analyst. 
Your task is to analyze the provided database schema, comprehend the posed question, and leverage the hint to identify which tables are needed to generate a SQL query for answering the question.

Database Schema Overview:
{DATABASE_SCHEMA}

This schema provides a detailed definition of the database's structure, including tables, their columns, primary keys, foreign keys, and any relevant details about relationships or constraints.
For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a critical hint to identify the tables that will be used in the SQL query.

Task:
Based on the database schema, question, and hint provided, your task is to determine the tables that should be used in the SQL query formulation. 
For each of the selected tables, explain why exactly it is necessary for answering the question. Your explanation should be logical and concise, demonstrating a clear understanding of the database schema, the question, and the hint."""

inputs = ["QUESTION", "HINT"]

output_format = 'response'

output_format_instructions = \
"""Please respond with a JSON object structured as follows:

```json
{{
  "chain_of_thought_reasoning": "Explanation of the logical analysis that led to the selection of the tables.",
  "table_names": ["Table1", "Table2", "Table3", ...]
}}
```

Note that you should choose all and only the tables that are necessary to write a SQL query that answers the question effectively.
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

    exec = LangChainLM('table_selection', semantic, opt_register=True)
    exec.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}
    runnable_exec = exec.as_runnable()
    registry[schema_string] = runnable_exec
    return runnable_exec | JsonOutputParser(pydantic_object=TableSelectionOutputParser)