from __future__ import annotations

from pydantic import BaseModel, Field


class ModifiedCodeSchema(BaseModel):
    modified_code: str = Field(
        ...,
        description='The code after applying the specified changes.',
        title='Modified Code',
    )
