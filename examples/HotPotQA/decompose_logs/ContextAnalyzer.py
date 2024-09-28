from __future__ import annotations

from pydantic import BaseModel, Field


class KeyPointsSchema(BaseModel):
    key_points: str = Field(
        ...,
        description='Summary of key points, facts, and relevant details from the context.',
        title='Key Points',
    )
