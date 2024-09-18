from __future__ import annotations

from pydantic import BaseModel, Field


class CorrectedCodeSchema(BaseModel):
    code: str = Field(..., description='Corrected Python code.', title='Code')
