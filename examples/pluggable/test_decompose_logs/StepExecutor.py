from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class StepExecutorOutput(BaseModel):
    step_outputs: List[str] = Field(
        ..., description='Outputs for each executed step', title='Step Outputs'
    )
