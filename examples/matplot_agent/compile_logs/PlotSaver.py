from __future__ import annotations

from pydantic import BaseModel, Field


class CodeSchema(BaseModel):
    code: str = Field(..., description='The final runnable Python code.', title='Code')
