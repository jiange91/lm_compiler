import importlib.util
from pathlib import Path
from compiler.llm import CogLM
from compiler.frontends.dspy.connector import as_predict, PredictCogLM
import dspy

from compiler.optimizer import (
  clear_registry,
  get_registered_opt_program_entry, 
  get_registered_opt_modules, 
  get_registered_opt_score_fn,
)

def load_module_from_path(module_path: str):
  path = Path(module_path)
  spec = importlib.util.spec_from_file_location(path.stem, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  print(get_registered_opt_modules())
  #print(module.__dict__)
  for k,v in module.__dict__.items():
    if isinstance(v, dspy.Module):
      print(k, type(v))
      print(v.__dict__)
      for name,predictor in v.__dict__.items():
        if isinstance(predictor, dspy.Predict):
          module.__dict__[k].__dict__[name] = PredictCogLM(predictor)
  #print(module.__dict__)
  print(get_registered_opt_modules())

module_path = 'plugin_example.py'

load_module_from_path(module_path)