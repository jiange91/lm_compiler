from __future__ import annotations

from pydantic import BaseModel, Field


class ExpendedQuerySchema(BaseModel):
    expended_query: str = Field(
        ...,
        description='Detailed instructions on how to write Python code.',
        title='Expended Query',
    )
