from __future__ import annotations

from pydantic import BaseModel, Field


class CodeSchema(BaseModel):
    code: str = Field(..., description='Generated Python code.', title='Code')
