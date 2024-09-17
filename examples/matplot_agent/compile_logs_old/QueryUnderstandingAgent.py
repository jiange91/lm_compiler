from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class ObjectivesSchema(BaseModel):
    objectives: List[str] = Field(
        ...,
        description='List of specific objectives to be achieved through Python code.',
        title='Objectives',
    )
