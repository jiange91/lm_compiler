from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class ChangeListSchema(BaseModel):
    change_list: str = Field(
        ...,
        description='Detailed list of specific changes to be made to the code.',
        title='Change List',
    )
