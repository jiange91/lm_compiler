from __future__ import annotations

from pydantic import BaseModel, Field


class VerifiedAnswerSchema(BaseModel):
    answer: str = Field(
        ...,
        description='Verified and validated answer to the question.',
        title='Answer',
    )
