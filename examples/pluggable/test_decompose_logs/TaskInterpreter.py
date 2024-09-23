from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class TaskInterpreterOutput(BaseModel):
    interpreted_task: str = Field(
        ..., description='Detailed description of the task', title='Interpreted Task'
    )
    ordered_steps: List[str] = Field(
        ...,
        description='List of steps in the order they should be followed',
        title='Ordered Steps',
    )
