from __future__ import annotations

from pydantic import BaseModel, Field


class ConfidenceDecisionSchema(BaseModel):
    confidence: int = Field(
        ...,
        description='A rating from 1 to 5 (low, medium, high, very high, absolute).',
        title='Confidence',
    )
    decision: str = Field(
        ...,
        description='A decision that has to be one of the following: Accept, Reject.',
        title='Decision',
    )
