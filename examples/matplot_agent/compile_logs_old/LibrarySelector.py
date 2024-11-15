from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LibrariesSchema(BaseModel):
    libraries: List[str] = Field(
        ..., description='List of selected Python libraries.', title='Libraries'
    )
