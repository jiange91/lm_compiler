from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionBreakdownSchema(BaseModel):
    question_breakdown: str = Field(
        ...,
        description='Breakdown of the question into its core components.',
        title='Question Breakdown',
    )
