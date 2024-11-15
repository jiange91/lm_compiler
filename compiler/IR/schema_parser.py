import importlib.util
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType
from pydantic import BaseModel, Field
from typing import Union

from datamodel_code_generator.parser.jsonschema import JsonSchemaParser


NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9]+")
UPPER_CAMEL_CASE = re.compile(r"[A-Z][a-zA-Z0-9]+")
LOWER_CAMEL_CASE = re.compile(r"[a-z][a-zA-Z0-9]+")

class BadJsonSchema(Exception):
    pass


def _to_camel_case(name: str) -> str:
    if any(NON_ALPHANUMERIC.finditer(name)):
        return "".join(term.lower().title() for term in NON_ALPHANUMERIC.split(name))
    if UPPER_CAMEL_CASE.match(name):
        return name
    if LOWER_CAMEL_CASE.match(name):
        return name[0].upper() + name[1:]
    raise BadJsonSchema(f"Unknown case used for {name}")


def _load_module_from_file(file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name=file_path.stem, location=str(file_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)
    return module

def json_schema_to_pydantic_model(json_schema: dict, file_path: str) -> type[BaseModel]:
    json_schema_as_str = json.dumps(json_schema)
    pydantic_models_as_str: str = JsonSchemaParser(json_schema_as_str).parse()
    
    module_file_path = Path(file_path).resolve()
    with open(module_file_path, "wb+") as f:
        f.write(pydantic_models_as_str.encode())

    module = _load_module_from_file(file_path=module_file_path)

    main_model_name = _to_camel_case(name=json_schema["title"])
    pydantic_model: type[BaseModel] = module.__dict__[main_model_name]
    return pydantic_model

def pydentic_model_repr(model: type[BaseModel]) -> str:
    """Get str representation of a Pydantic model
    
    Will return the class definition of the Pydantic model as a string.
    """
    pydantic_str = JsonSchemaParser(
        json.dumps(model.model_json_schema())
    ).parse(with_import=False)
    return pydantic_str

class InnerModel(BaseModel):
    """A nested model"""
    a: int = Field(description="An integer field")
    b: str = Field(description="A string field")
    
class ExampleModel(BaseModel):
    """An example output schema"""
    ms: list[InnerModel] = Field(description="A list of InnerModel")
    meta: dict[str, str] = Field(description="A dictionary of string to string")

example_output_json = """
```json
{
    "ms": [
        {"a": 1, "b": "b1"},
        {"a": 2, "b": "b2"}
    ],
    "meta": {"key1": "value1", "key2": "value2"}
}
```
"""

def get_pydantic_format_instruction(schema: type[BaseModel]):
    
    template = """\
Your answer should be formatted as a JSON instance that conforms to the output schema. The json instance will be used directly to instantiate the Pydantic model.

As an example, given the output schema:
{example_output_schema}

Your answer in this case should be formatted as follows:
{example_output_json}

Here's the real output schema for your reference:
{real_output_schema}

Please provide your answer in the correct json format accordingly. Especially make sure each field will respect the type and constraints defined in the schema.
Pay attention to the enum field in properties, do not generate answer that is not in the enum field if provided.
"""
    return template.format(
        example_output_schema=pydentic_model_repr(ExampleModel),
        example_output_json=example_output_json,
        real_output_schema=pydentic_model_repr(schema),
    )
