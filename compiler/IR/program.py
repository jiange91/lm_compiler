from typing import List, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import queue
from enum import Enum, auto

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

class Module:
    def __init__(self, name, kernel) -> None:
        self.name = name
        self.kernel = kernel
        self.children = []
        self.dependencies = []
        self.outputs = []
        self.status = None
    
    def forward(self, state: StatePool):
        raise NotImplementedError
    
    def __call__(self, state: StatePool):
        result = self.forward(state)
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
    
    def add_module(self, module: Module) -> None:
        self.modules.append(module)
    
    def add_edge(self, parent: Module, child: Module) -> None:
        parent.children.append(child)
        child.dependencies.append(parent)
        self.edges[parent].append(child)
    
    def reset_modules(self) -> None:
        for module in self.modules:
            module.clean()
    
    def run(self,
            state,
            start_from: Optional[Module] = None,
            stop_at: Optional[Module] = None):
        sorted_modules = self.sort()
        started = False
        for module in sorted_modules:
            if start_from is None or (module is start_from and not started):
                started = True
            if not started:
                module.status = ModuleStatus.SKIPPED
                continue
            deps = module.dependencies
            if deps is None or all(m.statis is ModuleStatus.SUCCESS for m in deps):
                module(state)
                answer = module.outputs[-1]
                if module == stop_at:
                    return module.outputs[-1]
            else:
                module.statis = ModuleStatus.SKIPPED
        return answer
    
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