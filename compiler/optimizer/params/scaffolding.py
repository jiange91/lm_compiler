from typing import Literal, Optional, Tuple
import uuid
import dataclasses
import heapq
import os
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

from compiler.IR.base import Module
from compiler.IR.program import Workflow
from compiler.IR.llm import LLMPredictor, Demonstration
from compiler.optimizer.params.common import EvolveType, ParamBase, ParamLevel, OptionBase, DynamicParamBase, IdentityOption
from compiler.optimizer.decompose import LMTaskDecompose
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, inspect_runnable
from compiler.optimizer.evaluation.evaluator import EvaluationResult, Evaluator
from compiler.optimizer.params.utils import dump_params, load_params

class LMScaffolding(ParamBase):
    level = ParamLevel.GRAPH
    
    def __init__(
        self,
        name: str,
        module_name: str,
    ):
        super().__init__(name, [IdentityOption()], 0, module_name)
    
    @classmethod
    def bootstrap(
        cls,
        workflow: Workflow,
        target_modules: Optional[list[str]] = None,
        log_path: Optional[str] = None,
    ):
        decomposer = LMTaskDecompose(workflow=workflow)
        