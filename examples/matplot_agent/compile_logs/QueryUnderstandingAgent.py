from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class RequirementsSummarySchema(BaseModel):
    requirements_summary: str = Field(
        ...,
        description="A clear and concise summary of the user's requirements.",
        title='Requirements Summary',
    )
