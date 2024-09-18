from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class LibrariesAndFunctionsSchema(BaseModel):
    libraries_and_functions: Dict[str, List[str]] = Field(
        ...,
        description='Dictionary of libraries and their corresponding functions.',
        title='Libraries and Functions',
    )
