from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(Enum):
    ae = 'ae'
    re = 're'
    ge = 'ge'
    accept = 'accept'


class KnowledgeRelevanceSchema(BaseModel):
    decision: Decision = Field(
        ...,
        description="Grade the knowledge as 're' if it is irrelevant to the question, otherwise pass to the next agent.",
        title='Decision',
    )
