from __future__ import annotations

from pydantic import BaseModel, Field


class SoundnessSchema(BaseModel):
    soundness: int = Field(
        ...,
        description='A rating from 1 to 4 (poor, fair, good, excellent).',
        title='Soundness',
    )
