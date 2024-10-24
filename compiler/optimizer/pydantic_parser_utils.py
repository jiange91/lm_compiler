import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from pydantic import BaseModel

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