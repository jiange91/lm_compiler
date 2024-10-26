import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))

from typing import Any
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import Demonstration, LMConfig
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from llm.parsers import SQLGenerationOutput, RawSqlOutputParser
from compiler.optimizer.params import ensemble
from langchain_core.runnables import chain


system_prompt = \
"""You are a data science expert.
Below, you are presented with a database schema and a question.
Your task is to understand the question, read the schema, and use the hint to pinpoint the specific columns to generate a valid SQLite query to answer the question.

<question>
A natural language question that requires querying a database to retrieve specific information.

<database_schema>
The schema offers an in-depth description of the database's architecture, detailing tables, columns, primary keys, foreign keys, and any pertinent information regarding relationships or constraints. Special attention should be given to the examples listed beside each column, as they directly hint at which columns are relevant to our query.

<hint>
The hint aims to direct your focus towards the specific elements of the database schema that are crucial for answering the question effectively.

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. For key phrases mentioned in the question, we have provided the most similar values within the columns denoted by "-- examples" in front of the corresponding column names. This is a crucial hint indicating the correct columns to use for your SQL query.
7. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
8. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
9. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
10. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns.

Priority should be given to columns that have been explicitly matched with examples relevant to the question's context.

Take a deep breath and think carefully to find the correct sqlite SQL query. If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
"""

inputs = ["QUESTION", "DATABASE_SCHEMA", "HINT"]

output_format = "sql_query"

output_format_instructions = \
"""
Please only provide a valid SQL query in a single string. Do not include any additional information or explanations.
"""

semantic = LangChainSemantic(
    system_prompt=system_prompt,
    inputs=inputs,
    output_format=output_format,
    output_format_instructions=output_format_instructions,
)
exec = LangChainLM('candidate_generation', semantic, opt_register=True)
raw_runnable_exec = exec.as_runnable() | RawSqlOutputParser()
exec.lm_config = LMConfig(
    provider='openai',
    model='gpt-4o-mini',
    cost_indicator=1.0,
    kwargs= {
        'temperature': 0.0,
    }
)

use_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
new_exec = use_ensemble.apply(exec)

ZeroShotCoT.direct_apply(new_exec.modules['candidate_generation_sampler_0'])
ZeroShotCoT.direct_apply(new_exec.modules['candidate_generation_sampler_1'])
PlanBefore.direct_apply(new_exec.modules['candidate_generation_sampler_2'])

exec.invoke = new_exec.invoke

@chain
def runnable_exec(input: dict):
    sql = raw_runnable_exec.invoke(input)
    return {"SQL": sql, "chain_of_thought_reasoning": ""}