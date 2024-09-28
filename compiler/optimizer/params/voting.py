from typing import Literal, Optional, Tuple
import uuid
import dataclasses
import heapq
import os
import sys
import json
import logging
from pathlib import Path
import copy


logger = logging.getLogger(__name__)

from compiler.IR.base import Module
from compiler.IR.program import Workflow
from compiler.IR.llm import LLMPredictor, Demonstration
from compiler.optimizer.params.common import EvolveType, ParamBase, ParamLevel, OptionBase, DynamicParamBase, IdentityOption, AddNewModuleImportInterface
from compiler.optimizer.decompose import LMTaskDecompose, StructuredAgentSystem
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, get_inspect_runnable
from compiler.optimizer.evaluation.evaluator import EvaluationResult, Evaluator
from compiler.optimizer.params.utils import dump_params, load_params
from compiler.optimizer.plugin import OptimizerSchema

class Voting(ParamBase, AddNewModuleImportInterface):
    level = ParamLevel.GRAPH
    
    def __init__(
        self,
        name: str,
        module_name: str = None,
        aggregator: str = 'llm_as_judge',
        default_identity: bool = True,
    ):
        if default_identity:
            options = [
                IdentityOption(),
            ]
        else:
            options = []
        super().__init__(name, options, 0, module_name)
        
class VotingOption(OptionBase):
    def __init__(self, name: str, aggregator: Module) -> None:
        super().__init__(name)
        self.aggregator = aggregator
    
    def apply(self, module: Module) -> Module:
        