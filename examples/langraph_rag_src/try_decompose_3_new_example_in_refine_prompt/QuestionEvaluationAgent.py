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
        description="Grade the answer as 'ae' if it does not address the question, otherwise pass to the next agent.",
        title='Decision',
    )
