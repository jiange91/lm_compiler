from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    re = 're'
    ge = 'ge'
    accept = 'accept'


class QuestionEvaluationSchema(BaseModel):
    decision: Decision = Field(
        ...,
        description='Decision on whether the answer addresses the question.',
        title='Decision',
    )
