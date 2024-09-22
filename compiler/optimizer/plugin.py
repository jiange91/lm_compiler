from typing import Callable, Iterable, Any
import runpy
import importlib
import logging
import os

logger = logging.getLogger(__name__)

from compiler.IR.program import Module, Workflow
from compiler.optimizer import (
    clear_registry,
    get_registered_opt_program_entry, 
    get_registered_opt_modules, 
    get_registered_opt_score_fn,
)
from compiler.optimizer import registry 

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
    def capture(cls, script_path: str) -> 'OptimizerSchema':
        # logger.info(f"initial modules: {id(registry._reg_opt_modules_)}")
        
        # NOTE: somehow modules un run_path might be imported before calling this function
        # so I force spawn now instead of fork to ensure these global registry are not shared and inherited when running the script
        # clear_registry()
        runpy.run_path(script_path)
        return cls(
            program=get_registered_opt_program_entry(),
            score_fn=get_registered_opt_score_fn(),
            opt_target_modules=get_registered_opt_modules(),
        )
