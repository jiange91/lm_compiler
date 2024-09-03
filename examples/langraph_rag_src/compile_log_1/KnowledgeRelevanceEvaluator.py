from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    re = 're'
    pass_ = 'pass'


class KnowledgeRelevanceDecision(BaseModel):
    decision: Decision = Field(
        ...,
        description="Decision on whether the knowledge is relevant to the user question, 're' or pass to next agent.",
        title='Decision',
    )
