from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class IdentifiedFunctionsSchema(BaseModel):
    identified_functions: List[str] = Field(
        ...,
        description='A list of Python functions suitable for the task.',
        title='Identified Functions',
    )
