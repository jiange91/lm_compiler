from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class RequirementsSchema(BaseModel):
    requirements: List[str] = Field(
        ...,
        description='List of requirements extracted from the query.',
        title='Requirements',
    )
