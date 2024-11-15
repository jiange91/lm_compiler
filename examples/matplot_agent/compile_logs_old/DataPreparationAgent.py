from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DataPreparationStepsSchema(BaseModel):
    data_preparation_steps: List[str] = Field(
        ...,
        description='List of steps to prepare and manipulate the data.',
        title='Data Preparation Steps',
    )
