from __future__ import annotations

from pydantic import BaseModel, Field


class DataPreparationStepsSchema(BaseModel):
    data_preparation_steps: str = Field(
        ...,
        description='Instructions on data preparation and manipulation.',
        title='Data Preparation Steps',
    )
