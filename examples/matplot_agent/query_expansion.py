SYSTEM_PROMPT = '''
According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. List all the appropriate libraries and pinpoint the correct library functions to call and set each parameter in every function call accordingly.

You should understand what the query's requirements are, and output step by step, detailed instructions on how to use python code to fulfill these requirements. Include what libraries to import, what library functions to call, how to set the parameters in each function correctly, how to prepare the data, how to manipulate the data so that it becomes appropriate for later functions to call etc,. Make sure the code to be executable and correctly generate the desired output in the user query. 
'''

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from langchain_core.pydantic_v1 import BaseModel, Field

class QueryExpansion(BaseModel):
    """Response from the query expansion task"""
    expanded_query: str = Field(
        description="details on how to perform the user query"
    )

query_expansion_semantic = LangChainSemantic(
    SYSTEM_PROMPT,
    ['query'],
    "expanded_query",
)

query_expansion_lm = LangChainLM('query expansion', query_expansion_semantic)