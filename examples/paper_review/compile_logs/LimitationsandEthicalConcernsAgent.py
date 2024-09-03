from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LimitationsEthicalConcernsSchema(BaseModel):
    limitations: List[str] = Field(
        ...,
        description='A list of limitations and potential negative societal impacts of the work.',
        title='Limitations',
    )
    ethical_concerns: bool = Field(
        ...,
        description='A boolean value (true or false) indicating whether there are ethical concerns.',
        title='Ethical Concerns',
    )
