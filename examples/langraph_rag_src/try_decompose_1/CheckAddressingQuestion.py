from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class AddressingDecision(Enum):
    ae = 'ae'
    valid = 'valid'


class AddressingDecisionSchema(BaseModel):
    addressing_decision: AddressingDecision = Field(
        ...,
        description="Decision on whether the answer addresses the question, 'ae' or 'valid'.",
        title='Addressing Decision',
    )
