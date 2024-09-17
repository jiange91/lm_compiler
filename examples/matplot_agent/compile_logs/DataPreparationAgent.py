from __future__ import annotations

from langchain_core.pydantic_v1 import BaseModel, Field


class DataPreparationStepsSchema(BaseModel):
    data_preparation_steps: str = Field(
        ...,
        description='Detailed instructions for data preparation.',
        title='Data Preparation Steps',
    )
