import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import Demonstration
from llm.parsers import ColumnFilteringOutput

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

inputs = ["QUESTION", "HINT", "COLUMN_PROFILE"]

output_format = ColumnFilteringOutput

output_format_instructions = \
"""Provide your answer in the following json format:

```json
{{
  "chain_of_thought_reasoning": "One line explanation of why or why not the column information is relevant to the question and the hint.",
  "is_column_information_relevant": "Yes" or "No"
}}
```

Only output a json as your response."""

demos = []

semantic = LangChainSemantic(
    system_prompt=system_prompt,
    inputs=inputs,
    output_format=output_format,
    output_format_instructions=output_format_instructions,
    demos=demos
)
exec = LangChainLM('column_filtering', semantic, opt_register=True)
exec.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}
runnable_exec = exec.as_runnable()