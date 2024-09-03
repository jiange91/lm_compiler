from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PassagesSchema(BaseModel):
    passages: List[str] = Field(..., description='A list of passages', title='Passages')
