from __future__ import annotations

from pydantic import BaseModel, Field


class PaperSummarySchema(BaseModel):
    summary: str = Field(
        ...,
        description='A summary of the paper content and its contributions.',
        title='Summary',
    )
