from __future__ import annotations

from pydantic import BaseModel, Field


class OverallScoreSchema(BaseModel):
    overall: int = Field(
        ...,
        description='A rating from 1 to 10 (very strong reject to award quality).',
        title='Overall',
    )
