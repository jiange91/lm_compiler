from __future__ import annotations

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field


class TaskListSchema(BaseModel):
    task_list: List[str] = Field(
        ..., description='A list of tasks to be performed.', title='Task List'
    )
