from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class ErrorExplanationSchema(BaseModel):
    error_explanation: str = Field(
        ...,
        description='Explanation of what is causing the error.',
        title='Error Explanation',
    )
