from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RelevanceDecision(Enum):
    re = 're'
    valid = 'valid'


class RelevanceDecisionSchema(BaseModel):
    relevance_decision: RelevanceDecision = Field(
        ...,
        description="Decision on the relevance of the knowledge to the question, 're' or 'valid'.",
        title='Relevance Decision',
    )
