from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    re = 're'
    ge = 'ge'
    accept = 'accept'


class GroundingEvaluationSchema(BaseModel):
    decision: Decision = Field(
        ...,
        description="Grade the answer as 'ge' if it is not grounded by the knowledge, otherwise grade it as 'accept'.",
        title='Decision',
    )
