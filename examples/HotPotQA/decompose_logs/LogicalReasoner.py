from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerSchema(BaseModel):
    answer: str = Field(
        ..., description='Constructed answer to the question.', title='Answer'
    )
