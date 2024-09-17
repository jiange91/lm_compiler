from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class ErrorExplanationSchema(BaseModel):
    error_explanation: str = Field(
        ...,
        description='Detailed explanation of the error and its likely cause.',
        title='Error Explanation',
    )
