from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class LibraryListSchema(BaseModel):
    library_list: List[str] = Field(
        ..., description='A list of libraries with explanations.', title='Library List'
    )
