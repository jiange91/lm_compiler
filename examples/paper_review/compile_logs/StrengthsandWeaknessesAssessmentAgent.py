from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class StrengthsWeaknessesSchema(BaseModel):
    strengths: List[str] = Field(
        ..., description='A list of strengths of the paper.', title='Strengths'
    )
    weaknesses: List[str] = Field(
        ..., description='A list of weaknesses of the paper.', title='Weaknesses'
    )
    originality: int = Field(
        ...,
        description='A rating from 1 to 4 (low, medium, high, very high)',
        title='Originality',
    )
    quality: int = Field(
        ...,
        description='A rating from 1 to 4 (low, medium, high, very high)',
        title='Quality',
    )
    clarity: int = Field(
        ...,
        description='A rating from 1 to 4 (low, medium, high, very high).',
        title='Clarity',
    )
    significance: int = Field(
        ...,
        description='A rating from 1 to 4 (low, medium, high, very high).',
        title='Significance',
    )
