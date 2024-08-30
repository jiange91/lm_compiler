from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PassagesSchema(BaseModel):
    passages: List[str] = Field(
        ..., description='Passages generated for the sub-topics', title='Passages'
    )
