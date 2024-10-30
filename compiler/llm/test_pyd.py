from pydantic import BaseModel, create_model
from typing import Any, Dict, Type

def generate_pydantic_model(model_name: str, fields: Dict[str, Type[Any]]) -> Type[BaseModel]:
    # Generate a dynamic Pydantic model using create_model
    return create_model(model_name, **{name: (field_type, ...) for name, field_type in fields.items()})

# Example usage
fields = {
    'name': str,
    'age': int,
    'is_student': bool
}

# Create the Pydantic model
PersonModel = generate_pydantic_model('PersonModel', fields)

# Instantiate and validate data with the model
person_instance = PersonModel(name="Alice", age=30, is_student=True)

print(isinstance(person_instance, BaseModel))  # Output: True