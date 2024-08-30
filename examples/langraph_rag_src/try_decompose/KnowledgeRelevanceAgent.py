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
        description='Decision on the relevance of the knowledge to the question.',
        title='Decision',
    )
