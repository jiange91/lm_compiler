from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ge = 'ge'
    accept = 'accept'


class GroundingDecision(BaseModel):
    decision: Decision = Field(
        ...,
        description="Decision on whether the answer is grounded by the provided knowledge, 'ge' or 'accept'.",
        title='Decision',
    )
