from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class IdentifiedLibrariesSchema(BaseModel):
    identified_libraries: List[str] = Field(
        ...,
        description='A list of Python libraries suitable for the task.',
        title='Identified Libraries',
    )
