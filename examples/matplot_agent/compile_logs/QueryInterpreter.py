from __future__ import annotations

from pydantic import BaseModel, Field


class RequirementsSchema(BaseModel):
    requirements: str = Field(
        ...,
        description='List of requirements and tasks to be performed.',
        title='Requirements',
    )
