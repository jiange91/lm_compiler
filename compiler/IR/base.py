from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable
import inspect
import time
import logging
import copy
import threading
from graphviz import Digraph

from compiler.IR.utils import get_function_kwargs

logger = logging.getLogger(__name__)
class State:
    def __init__(self, version_id, data, is_static) -> None:
        self.version_id = version_id # currently not used
        self.data = data
        self.is_static = is_static # if True, the state will not be updated anymore

class StatePool:
    def __init__(self):
        self.states: dict[str, list[State]] = defaultdict(list)
        
    def news(self, key: str, default = None):
        if key not in self.states or not self.states[key]:
            if default is None:
                raise ValueError(f"Key {key} not found in state")
            return default
        return self.states[key][-1].data
    
    def init(self, kvs):
        self.publish(kvs, is_static = True, version_id = 0)
    
    def publish(self, kvs, is_static, version_id):
        for key, value in kvs.items():
            self.states[key].append(State(version_id, value, is_static))
    
    def all_news(self, fields = None, excludes = None):
        report = {}
        for key in self.states:
            if fields is not None and key not in fields:
                continue
            if excludes is not None and key in excludes:
                continue
            report[key] = self.news(key)
        return report
    
    def history(self, key: str):
        if key not in self.states:
            raise ValueError(f"Key {key} not found in state")
        hist = []
        for state in self.states[key]:
            hist.append(state.data)
        return hist

    def all_history(self, fields = None, excludes = None):
        report = {}
        for key in self.states:
            if fields is not None and key not in fields:
                continue
            if excludes is not None and key in excludes:
                continue
            report[key] = self.history(key)
        return report
    
    def dump(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError

@dataclass
class Context:
    calling_module: 'Module'
    predecessor: str
    invoke_time: int # start from 1

def hint_possible_destinations(dests: list[str]):
    def hinter(func):
        func._possible_destinations = dests
        return func
    return hinter


class ModuleStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()

class ModuleIterface(ABC):
    @abstractmethod
    def __call__(self, statep: StatePool):
        ...
    
    @abstractmethod
    def reset(self):
        """clear metadata for new run
        """
        ...

class Module(ModuleIterface):
    def __init__(self, name, kernel) -> None:
        self.name = name
        self.kernel = kernel
        self.outputs = []
        self.exec_times = []
        self.status = None
        self.is_static = False
        self.version_id = 0
        self.enclosing_module: 'Module' = None
        if kernel is not None:
            self.input_fields, self.defaults = get_function_kwargs(kernel)
        else:
            self.input_fields, self.defaults = [], {}
        self.on_signature_generation()
        logger.debug(f"Module {name} kernel has input fields {self.input_fields}")
    
    def reset(self):
        self.outputs = []
        self.exec_times = []
        self.status = None
        self.version_id = 0
    
    def forward(self, **kwargs):
        raise NotImplementedError
    
    def get_immediate_enclosing_module(self):
        """return the immediate enclosing module of this module, None if this module is not in any module
        """
        return self.enclosing_module
    
    def on_signature_generation(self):
        """
        allows each child class to modify the input fields and defaults
        """
        pass

    def __call__(self, statep: StatePool):
        for field in self.input_fields:
            if field not in self.defaults and field not in statep.states:
                raise ValueError(f"Missing field {field} in state when calling {self.name}, available fields: {statep.states.keys()}")
        kargs = {field: statep.news(field) for field in statep.states if field in self.input_fields}
                
        # time the execution
        start = time.perf_counter()
        result = self.forward(**kargs)
        dur = time.perf_counter() - start
        result_snapshot = copy.deepcopy(result)
        statep.publish(result_snapshot, self.version_id, self.is_static)
        self.outputs.append(result_snapshot)
        # update metadata
        self.exec_times.append(dur)
        self.status = ModuleStatus.SUCCESS
        self.version_id += 1
    
class ComposibleModuleInterface(ABC):
    @abstractmethod
    def immediate_submodules(self) -> List[Module]:
        pass
    
    @abstractmethod
    def _visualize(self, dot: Digraph):
        pass

    def sub_module_validation(self, module: Module, reset_parent: bool):
        """
        Example: 
        ```python
        a = Module('a', None)
        b = Workflow('b'); b.add_module(a)
        c = Workflow('c')
        
        # YES
        c.add_module(b)
        # NO
        c.add_module(a)
        ```
        """
        if not reset_parent and (parent := module.enclosing_module) is not None:
            raise ValueError(f"Source module {module.name} already registered in {parent.name}, please avoid adding the same module to multiple modules")
    
    @abstractmethod
    def replace_node_handler(self, old_node: Module, new_node_in: Module, new_node_out: Module) -> bool:
        pass

    def replace_node(self, old_node: Module, new_node_in: Module, new_node_out: Module) -> bool:
        """Replace the old node with the new node
        
        Please register the new node to the same parent module as the old node before calling this method.
        
        Replace all old_node's occurrences as a predecessor with new_node_out
        
        Replace all old_node's occurrences as a successor with new_node_in
        
        new_node_in and new_node_out can be the same module, equivalent to replacing the old node with another node
        
        If old_node not found in the immediate submodules, will not recursively call the replace_node method for the submodules
        """
        submodules = self.immediate_submodules()
        if old_node not in submodules:
            logger.warning(f"Node {old_node.name} not found in {self.name}")
            return False
        
        return self.replace_node_handler(old_node, new_node_in, new_node_out)
            