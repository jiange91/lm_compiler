from __future__ import annotations

from pydantic import BaseModel, Field


class CodePlanSchema(BaseModel):
    code_plan: str = Field(
        ..., description='Detailed plan for the Python code.', title='Code Plan'
    )
