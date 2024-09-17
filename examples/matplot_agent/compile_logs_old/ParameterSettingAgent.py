from __future__ import annotations

from typing import Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class FunctionParameters(BaseModel):
    parameters: Optional[Dict[str, str]] = None


class FunctionParametersSchema(BaseModel):
    function_parameters: Dict[str, FunctionParameters] = Field(
        ...,
        description='Dictionary of functions and their parameter settings.',
        title='Function Parameters',
    )
