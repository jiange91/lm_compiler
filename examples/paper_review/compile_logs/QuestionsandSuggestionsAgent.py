from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class QuestionsSchema(BaseModel):
    questions: List[str] = Field(
        ...,
        description='A list of clarifying questions to be answered by the paper authors.',
        title='Questions',
    )
