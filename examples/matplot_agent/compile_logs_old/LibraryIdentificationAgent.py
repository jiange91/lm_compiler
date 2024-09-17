from __future__ import annotations

from typing import Dict, List

from langchain_core.pydantic_v1 import BaseModel, Field


class LibrariesAndFunctionsSchema(BaseModel):
    libraries_and_functions: Dict[str, List[str]] = Field(
        ...,
        description='Dictionary of libraries and their corresponding functions.',
        title='Libraries and Functions',
    )
