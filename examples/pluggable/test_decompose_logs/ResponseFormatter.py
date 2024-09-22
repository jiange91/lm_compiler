from __future__ import annotations

from pydantic import BaseModel, Field


class ResponseFormatterOutput(BaseModel):
    whatever: str = Field(
        ...,
        description='Formatted final response based on the execution report',
        title='Final Response',
    )
