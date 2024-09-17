from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class CodeSchema(BaseModel):
    code: str = Field(..., description='The modified Python code.', title='Code')
