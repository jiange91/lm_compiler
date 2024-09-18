from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ChangesSchema(BaseModel):
    changes: List[str] = Field(
        ...,
        description='List of specific changes to be made to the code.',
        title='Changes',
    )
