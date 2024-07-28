from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import queue
from enum import Enum, auto
import inspect

import logging

logger = logging.getLogger(__name__)

class StatePool:
    def __init__(self):
        self.state = defaultdict(list)
        
    def news(self, key: str, default = None):
        if key not in self.state or not self.state[key]:
            if default is None:
                raise ValueError(f"Key {key} not found in state")
            return default
        return self.state[key][-1]
    
    def publish(self, kvs):
        for key, value in kvs.items():
            self.state[key].append(value)
    
    def dump(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError

class ModuleStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()

def get_function_kwargs(func):
    signature = inspect.signature(func)
    input_fields = []
    defaults = {}
    for name, param in signature.parameters.items():
        if name == 'self':
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL or param.kind == inspect.Parameter.VAR_KEYWORD:
            raise ValueError("Only support keyword arguments")
        input_fields.append(name)
        if param.default != inspect.Parameter.empty:
            defaults[name] = param.default
    return input_fields, defaults

class Module:
    def __init__(self, name, kernel) -> None:
        self.name = name
        self.kernel = kernel
        self.children: List[Module] = []
        self.dependencies: list[Module] = []
        self.outputs = []
        self.status = None
        self.input_fields, self.defaults = get_function_kwargs(kernel)
        logger.debug(f"Module {name} kernel has input fields {self.input_fields}")
    
    def forward(self, **kwargs):
        raise NotImplementedError
    
    def __call__(self, state: StatePool):
        kargs = {}
        for field in self.input_fields:
            if field not in self.defaults and field not in state.state:
                raise ValueError(f"Missing field {field} in state")
            if field in state.state:
                kargs[field] = state.news(field)
        result = self.forward(**kargs)
        self.outputs.append(result)
        state.publish(result)
        self.statis = ModuleStatus.SUCCESS

    def clean(self):
        self.outputs = []

    
class Workflow:
    def __init__(self) -> None:
        self.dest: Module = None
        self.modules: List[Module] = []
        self.edges: dict[Module, List[Module]] = defaultdict(list)
        self.states: StatePool = None
        self.exit_point: Tuple[Module, str] = None
    
    def add_module(self, module: Module) -> None:
        self.modules.append(module)
    
    def add_edge(self, parent: Module, child: Module) -> None:
        parent.children.append(child)
        child.dependencies.append(parent)
        self.edges[parent].append(child)
    
    def set_exit_point(self, module: Module, field: str) -> None:
        self.exit_point = (module, field)
    
    def reset_modules(self) -> None:
        for module in self.modules:
            module.clean()
    
    def run(self,
            state,
            start_from: Optional[Module] = None,
            stop_before: Optional[Module] = None):
        sorted_modules = self.sort()
        started = False
        answer = None
        for module in sorted_modules:
            if start_from is None or (module is start_from and not started):
                started = True
            if not started:
                module.status = ModuleStatus.SKIPPED
                continue
            if module == stop_before:
                return answer
            deps = module.dependencies
            if deps is None or all(m.statis is ModuleStatus.SUCCESS for m in deps):
                module(state)
                answer = module.outputs[-1]
            else:
                module.statis = ModuleStatus.FAILED
                raise ValueError(f"Module {module.name} failed to run due to dependencies")
        return self.exit_point[0].outputs[-1][self.exit_point[1]]
    
    def sort(self, predicate: Optional[callable] = None) -> List[Module]:
        visited = {v: False for v in self.modules}
        stack = []
        
        def dfs(v):
            visited[v] = True
            for child in self.edges[v]:
                if not visited[child]:
                    dfs(child)
            stack.append(v)
        
        for v in self.modules:
            if not visited[v]:
                dfs(v)
        if predicate is None:
            return list(reversed(stack))
        return [v for v in reversed(stack) if predicate(v)]