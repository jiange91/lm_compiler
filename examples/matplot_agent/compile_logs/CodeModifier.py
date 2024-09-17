from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class ModifiedCodeSchema(BaseModel):
    code: str = Field(..., description='The final modified Python code.', title='Code')
