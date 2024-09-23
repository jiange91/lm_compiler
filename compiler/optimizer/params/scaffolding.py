from typing import Literal, Optional, Tuple
import uuid
import dataclasses
import heapq
import os
import sys
import json
import logging
from pathlib import Path


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

class LMScaffolding(ParamBase, AddNewModuleImportInterface):
    level = ParamLevel.GRAPH
    
    def __init__(
        self,
        name: str,
        module_name: str,
        log_dir: str,
        new_agent_systems: list[StructuredAgentSystem] = [],
    ):
        self.log_dir = log_dir
        options = [IdentityOption()]
        for i, new_sys in enumerate(new_agent_systems):
            options.append(
                DecomposeOption(f'Decompose_{module_name}_option_{i}', new_sys, log_dir)
            )
        super().__init__(name, options, 0, module_name)
    
    @classmethod
    def bootstrap(
        cls,
        workflow: Optional[Workflow] = None,
        lm_modules: Optional[list[LLMPredictor]] = None,
        decompose_threshold: int = 4,
        target_modules: Optional[list[str]] = None,
        log_dir: str = 'task_decompose_logs',
    ):
        decomposer = LMTaskDecompose(workflow=workflow, lm_modules=lm_modules)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        decomposer.prepare_decompose_metadata(target_modules, decompose_threshold, log_dir)
        decomposer.finalize_decomposition(target_modules, log_dir)
        
        params = []
        for module_name, new_system in decomposer.lm_2_final_system.items():
            params.append(cls(f'Scaffold_{module_name}', module_name, log_dir, [new_system]))
        return params

    @classmethod
    def bootstrap_from_source(
        cls,
        script_path: str,
        script_args: list[str] = [],
        decompose_threshold: int = 4,
        target_modules: Optional[list[str]] = None,
        log_dir: str = 'task_decompose_logs',
    ):
        dir = os.path.dirname(script_path)
        if dir not in sys.path:
            sys.path.insert(0, dir)
        sys.argv = [script_path] + script_args
        schema = OptimizerSchema.capture(script_path)
        lm_modules = schema.opt_target_modules
        return cls.bootstrap(
            lm_modules=lm_modules,
            decompose_threshold=decompose_threshold,
            target_modules=target_modules,
            log_dir=log_dir,
        )

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, log_dir = data['name'], data['module_name'], data['log_dir']
        param = cls(name, module_name, log_dir)
        
        loaded_options = data['options']
        loaded_options.pop('Identity', None)
        loaded_options = {name: DecomposeOption.from_dict(option) for name, option in loaded_options.items()}
        param.options.update(loaded_options)
        
        return param

    def to_dict(self):
        base = super().to_dict()
        base['log_dir'] = self.log_dir
        return base
    
    def get_python_paths(self) -> list[str]:
        return [self.log_dir]

class DecomposeOption(OptionBase):
    def __init__(self, name: str, new_system: StructuredAgentSystem, log_dir: str):
        super().__init__(name)
        self.new_system = new_system
        self.log_dir = log_dir
    
    def apply(self, module: LLMPredictor) -> Module:
        new_agent = LMTaskDecompose.materialize_decomposition(
            lm=module,
            new_agents=self.new_system,
            default_lm_config=None,
            log_dir=self.log_dir,
        )
        return new_agent

    def to_dict(self):
        base = super().to_dict()
        base['new_system'] = self.new_system.model_dump()
        base['log_dir'] = self.log_dir
        return base
    
    @classmethod
    def from_dict(cls, data: dict):
        name = data['name']
        new_system = StructuredAgentSystem.model_validate(data['new_system'])
        log_dir = data['log_dir']
        return cls(name, new_system, log_dir)