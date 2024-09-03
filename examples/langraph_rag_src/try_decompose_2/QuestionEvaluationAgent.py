from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    pass_ = 'pass'


class QuestionEvaluationSchema(BaseModel):
    decision: Decision = Field(
        ...,
        description="Decision on whether the answer addresses the question, 'ae' or pass to next agent.",
        title='Decision',
    )
