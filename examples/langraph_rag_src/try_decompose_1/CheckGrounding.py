from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GroundingDecision(Enum):
    ge = 'ge'
    valid = 'valid'


class GroundingDecisionSchema(BaseModel):
    grounding_decision: GroundingDecision = Field(
        ...,
        description="Decision on whether the answer is grounded in the knowledge, 'ge' or 'valid'.",
        title='Grounding Decision',
    )
