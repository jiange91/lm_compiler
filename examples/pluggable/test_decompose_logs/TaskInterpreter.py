from __future__ import annotations

from pydantic import BaseModel, Field


class TaskInterpreterOutput(BaseModel):
    interpreted_task: str = Field(
        ..., description='Detailed breakdown of the task', title='Interpreted Task'
    )
    interpreted_steps: str = Field(
        ..., description='Detailed breakdown of the steps', title='Interpreted Steps'
    )
