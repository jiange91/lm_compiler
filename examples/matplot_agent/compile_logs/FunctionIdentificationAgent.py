from __future__ import annotations

from pydantic import BaseModel, Field


class IdentifiedFunctionsSchema(BaseModel):
    identified_functions: str = Field(
        ...,
        description='List of Python functions and their parameters.',
        title='Identified Functions',
    )
