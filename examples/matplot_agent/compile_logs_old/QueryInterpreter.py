from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class RequirementsSchema(BaseModel):
    requirements: List[str] = Field(
        ...,
        description='List of requirements extracted from the query.',
        title='Requirements',
    )
