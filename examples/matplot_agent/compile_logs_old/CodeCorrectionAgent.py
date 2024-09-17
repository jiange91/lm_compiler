from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class CorrectedCodeSchema(BaseModel):
    code: str = Field(..., description='Corrected Python code.', title='Code')
