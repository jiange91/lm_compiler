from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    re = 're'
    ge = 'ge'
    accept = 'accept'


class FinalDecisionSchema(BaseModel):
    decision: Decision = Field(
        ...,
        description="Final decision, 'ae', 're', 'ge', or 'accept'.",
        title='Decision',
    )
