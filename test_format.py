from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Dict, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json

class ComplexityEstimation(BaseModel):
    """complexity of each agent"""
    score: int = Field(
        description="complexity score of the agent"
    )
    rationale: str = Field(
        description="rationale for the complexity score"
    )

class ComplexityList(BaseModel):
    """complexity of all agents"""
    es: List[ComplexityEstimation] = Field(
        description="list of complexity descriptions"
    )
    
print(json.dumps(json.loads(ComplexityList.schema_json()), indent=4))
parser = JsonOutputParser(pydantic_object=ComplexityList)
print(parser.get_format_instructions())