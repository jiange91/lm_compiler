from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorExplanationSchema(BaseModel):
    error_explanation: str = Field(
        ...,
        description='Explanation of what is causing the error.',
        title='Error Explanation',
    )
