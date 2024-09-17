
from typing import Dict, List

from pydantic import BaseModel, Field
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

print(json.dumps(ComplexityList.model_json_schema(), indent=4))

a = """
def next_agent(decision): 
    return ['END'] if decision == 'fail' else ['ProductManagerAgent']
"""
print(repr(a))