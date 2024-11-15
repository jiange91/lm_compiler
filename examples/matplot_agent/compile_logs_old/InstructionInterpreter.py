from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ChangesSchema(BaseModel):
    changes: List[str] = Field(
        ..., description='Detailed list of changes or improvements.', title='Changes'
    )
