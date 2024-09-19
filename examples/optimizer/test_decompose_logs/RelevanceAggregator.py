from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentFilter(BaseModel):
    decision: str = Field(..., description='yes or no', title='Decision')
