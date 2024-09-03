from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    pass_ = 'pass'


class QuestionRelevanceDecision(BaseModel):
    decision: Decision = Field(
        ...,
        description="Decision on whether the answer addresses the user question, 'ae' or pass to next agent.",
        title='Decision',
    )
