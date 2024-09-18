from __future__ import annotations

from pydantic import BaseModel, Field


class ClarifiedRequirementsSchema(BaseModel):
    clarified_requirements: str = Field(
        ...,
        description='Detailed breakdown of the user query into specific tasks or goals.',
        title='Clarified Requirements',
    )
