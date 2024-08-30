from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SubQuestionsSchema(BaseModel):
    sub_questions: List[str] = Field(
        ..., description='A list of sub-topics', title='Sub Questions'
    )
