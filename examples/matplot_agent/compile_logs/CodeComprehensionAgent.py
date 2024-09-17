from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class CodeExplanationSchema(BaseModel):
    code_explanation: str = Field(
        ...,
        description="Detailed explanation of the code's purpose and functionality.",
        title='Code Explanation',
    )
