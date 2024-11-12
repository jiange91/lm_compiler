import importlib.util
from typing import Callable, Iterable, Any
import runpy
import importlib
import logging
import os
from pathlib import Path

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
        score_fn: Callable[[Any, Any], float],
        opt_target_modules: list[Module],
    ):
        self.program = program
        self.score_fn = score_fn
        self.opt_target_modules = opt_target_modules
        # logger.info(f"modules cap: {opt_target_modules}")

    @classmethod
    def capture(cls, script_path: str, evaluator_path: str) -> 'OptimizerSchema':
        clear_registry(),
        capture_module_from_fs(script_path)
        capture_module_from_fs(evaluator_path)
        schema = cls(
            program=get_registered_opt_program_entry(),
            score_fn=get_registered_opt_score_fn(),
            opt_target_modules=get_registered_opt_modules(),
        )
        return schema

def capture_module_from_fs(module_path: str):
    logger.debug(f"obtain module at: {module_path}")
    path = Path(module_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    