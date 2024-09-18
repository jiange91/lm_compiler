from __future__ import annotations

from pydantic import BaseModel, Field


class IdentifiedLibrariesSchema(BaseModel):
    identified_libraries: str = Field(
        ...,
        description='List of Python libraries and their roles.',
        title='Identified Libraries',
    )
