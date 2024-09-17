from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class ExpendedQuerySchema(BaseModel):
    expended_query: str = Field(
        ...,
        description='Detailed instructions on how to write Python code.',
        title='Expended Query',
    )
