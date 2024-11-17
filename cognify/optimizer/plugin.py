import importlib.util
from typing import Callable, Iterable, Any
import runpy
import importlib
import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from cognify.graph.program import Module, Workflow
from cognify.optimizer import (
    clear_registry,
    get_registered_opt_program_entry, 
    get_registered_opt_modules, 
    get_registered_opt_score_fn,
)

class OptimizerSchema:
    def __init__(
        self,
        program: Callable[[Any], Any],
        opt_target_modules: list[Module],
    ):
        self.program = program
        self.opt_target_modules = opt_target_modules
        # logger.info(f"modules cap: {opt_target_modules}")

    @classmethod
    def capture(cls, script_path: str) -> 'OptimizerSchema':
        clear_registry(),
        capture_module_from_fs(script_path)
        schema = cls(
            program=get_registered_opt_program_entry(),
            opt_target_modules=get_registered_opt_modules(),
        )
        return schema

def capture_module_from_fs(module_path: str):
    logger.debug(f"obtain module at: {module_path}")
    
    try:
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load module from {module_path}")
        raise
    
    
class EntryBase:
    def __init__(self):
        pass
        
    def _get_all_targets(self):
        """Iteratively finds all targets
        """
        found_modules = []
        stack = [self]  # Initialize stack with the root Entry instance

        while stack:
            current_obj = stack.pop()

            for attr_value in current_obj.__dict__.values():
                if isinstance(attr_value, Module):
                    # found a registered Module
                    if attr_value.opt_target:
                        found_modules.append(attr_value)
                        
                elif isinstance(attr_value, dict):
                    stack.extend(attr_value.values())
                    
                elif isinstance(attr_value, (list, tuple, set)):
                    stack.extend(attr_value)
                    
                elif isinstance(attr_value, EntryBase):
                    stack.append(attr_value)

        return found_modules

class Entry(EntryBase, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        ...
        
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)