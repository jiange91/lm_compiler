from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Plan(BaseModel):
    steps: List[str] = Field(
        ...,
        description='different steps to follow, should be in sorted order',
        title='Steps',
    )
