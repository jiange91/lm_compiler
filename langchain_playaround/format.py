import inspect
from pydantic import BaseModel, Field
from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
import json

# Define the Pydantic models
class Score(BaseModel):
    """Ratings of a research paper"""
    presentation: int = Field(
        description="Presentation of the research paper"
    )
    novelty: int = Field(
        description="Novelty of the research paper"
    )

class PaperReviews(BaseModel):
    """Reviews a list of research papers"""
    scores: dict[str, Score] = Field(
        description="dictionary of paper title to its score"
    )
    
    
pydantic_str = JsonSchemaParser(json.dumps(PaperReviews.model_json_schema())).parse(with_import=False)
print(pydantic_str)