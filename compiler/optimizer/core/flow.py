import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Type
import copy
import logging
from collections import defaultdict
import uuid
from dataclasses import dataclass, field

from compiler.IR.program import Workflow, Module, StatePool
from compiler.optimizer.params.common import ParamBase
from compiler.optimizer.plugin import OptimizerSchema

logger = logging.getLogger(__name__)


class ModuleTransformTrace:
    """
    This represents the morphing of the workflow
    Example:
        orignal workflow: [a, b, c]
        valid module_name_paths: 
            a -> [a1, a2]
            a1 -> [a3, a4]
            c -> [c1]
        
        meaning original module A is now replaced by A2, A3, A4, upon which optimization will be performed
    """
    def __init__(self, ori_module_dict) -> None:
        self.ori_module_dict: dict[str, Type[Module]] = ori_module_dict
        # old_name to new_name
        self.module_name_paths: dict[str, str] = {}
        # level_name -> module_name -> [(param_name, option_name)]
        self.aggregated_proposals: dict[str, dict[str, list[tuple[str, str]]]] = {}
        self.flattened_name_paths: dict[str, str] = {}
    
    def add_mapping(self, ori_name: str, new_name: str):
        if ori_name in self.module_name_paths:
            raise ValueError(f'{ori_name} already been changed')
        self.module_name_paths[ori_name] = new_name
    
    def register_proposal(self, level_name: str, proposal: list[tuple[str, str, str]]):
        if level_name not in self.aggregated_proposals:
            self.aggregated_proposals[level_name] = defaultdict(list)
        for module_name, param_name, option_name in proposal:
            self.aggregated_proposals[level_name][module_name].append((param_name, option_name))
    
    def mflatten(self):
        """flatten the mapping
        
        Example:
            original: 
                a -> a1
                a1 -> a2
            after:
                a -> a2
        """
        def is_cyclic_util(graph, v, visited, rec_stack):
            visited[v] = True
            rec_stack[v] = True
            
            if v in graph:
                neighbor = graph[v]
                if not visited[neighbor]:
                    if is_cyclic_util(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack[neighbor]:
                    return True
                
            rec_stack[v] = False
            return False

        visited = defaultdict(False.__bool__)
        rec_stack = defaultdict(False.__bool__)
        for key in self.module_name_paths:
            if not visited[key]:
                if is_cyclic_util(self.module_name_paths, key, visited, rec_stack):
                    raise ValueError('Cyclic mapping')
        
        result: dict[str, str] = {}
        # use self.ori_module_dict to exclude sub-module mappings
        for ori in self.ori_module_dict:
            new_m_name = ori
            while new_m_name in self.module_name_paths:
                new_m_name = self.module_name_paths[new_m_name]
            result[ori] = new_m_name
        self.flattened_name_paths = result
    
    def get_derivatives_of_same_type(self, new_module: Module) -> Tuple[str, list[Module]]:
        """
        NOTE: call mflatten before this
        find the parent module of the given new module, will search recursively in the new_module
        
        return old module name and names of new derivatives of the same type as the old module
        """
        if new_module.name in self.ori_module_dict:
            return (new_module.name, [new_module])
        for ori_name, new_name in self.flattened_name_paths.items():
            if new_module.name == new_name:
                derivatives = Module.all_of_type([new_module], self.ori_module_dict[ori_name])
                return (ori_name, derivatives)
        return (new_module.name, [new_module])

    def eq_transform_path(self, other: dict[str, str]) -> bool:
        if self.module_name_paths.keys() != other.keys():
                return False
        for old_name, new_name in self.module_name_paths.items():
            if new_name != other[old_name]:
                return False
        return True

@dataclass
class OptConfig:
    n_trials: int
    throughput: int = field(default=2)
    log_dir: str = field(default=None)
    evolve_interval: int = field(default=2)
    opt_log_path: str = field(default=None)
    param_save_path: str = field(default=None)
    frugal_eval_cost: bool = field(default=True)
    
    
    def finalize(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if self.opt_log_path is None:
            self.opt_log_path = os.path.join(self.log_dir, 'opt_logs.json')
        if self.param_save_path is None:
            self.param_save_path = os.path.join(self.log_dir, 'opt_params.json')
    
    def update(self, other: 'OptConfig'):
        # for all not None fields in other, update self
        for key, value in other.__dict__.items():
            if value is not None:
                setattr(self, key, value)

@dataclass
class TopDownInformation:
    """
    Information that is passed from the top level to the lower level
    """
    # optimization config for current level
    opt_config: OptConfig
    
    # optimization meta inherit from the previous level
    all_params: Optional[dict[str, ParamBase]]
    module_ttrace: Optional[ModuleTransformTrace]
    current_module_pool: Optional[dict[str, Module]]
    
    # optimization configs
    script_path: str
    script_args: Optional[list[str]]
    other_python_paths: Optional[list[str]]
    
    def initialize(self):
        self.opt_config.finalize()
        self.all_params = self.all_params or {}
        self.script_args = self.script_args or []
        self.other_python_paths = self.other_python_paths or []
        
        if self.current_module_pool is None:
            dir = os.path.dirname(self.script_path)
            if dir not in sys.path:
                sys.path.insert(0, dir)
            sys.argv = [self.script_path] + self.script_args
            schema = OptimizerSchema.capture(self.script_path)
            self.current_module_pool = {m.name: m for m in schema.opt_target_modules}
        
        if self.module_ttrace is None:
            name_2_type = {m.name: type(m) for m in self.current_module_pool.values()}
            self.module_ttrace = ModuleTransformTrace(name_2_type)
        self.module_ttrace.mflatten()
    
        
class TrialLog:
    def __init__(
        self,
        params: dict[str, any],
        bo_trial_id: int,
        id: str = None,
        score: float = 0.0,
        price: float = 0.0,
        eval_cost: float = 0.0,
    ):
        self.id: str = id or uuid.uuid4().hex
        self.params = params
        self.bo_trial_id = bo_trial_id
        self.score = score
        self.price = price
        self.eval_cost = eval_cost
    
    def to_dict(self):
        return {
            'id': self.id,
            'bo_trial_id': self.bo_trial_id,
            'params': self.params,
            'score': self.score,
            'price': self.price,
            'eval_cost': self.eval_cost,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)