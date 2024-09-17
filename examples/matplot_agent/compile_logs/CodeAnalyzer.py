from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class CodeSummarySchema(BaseModel):
    code_summary: str = Field(
        ...,
        description="Detailed summary of the code's functionality.",
        title='Code Summary',
    )
