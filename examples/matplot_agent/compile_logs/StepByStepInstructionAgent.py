from __future__ import annotations

from pydantic import BaseModel, Field


class ExpandedQuerySchema(BaseModel):
    expanded_query: str = Field(
        ...,
        description='Detailed instructions on how to write Python code.',
        title='Expanded Query',
    )
