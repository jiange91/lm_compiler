from typing import List, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
import queue

class StatePool:
    def __init__(self):
        self.state = defaultdict(list)
        
    def news(self, key: str):
        return self.state[key][-1]
    
    def publish(self, kvs):
        for key, value in kvs.items():
            self.state[key].append(value)
    
    def dump(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError
    

class Module:
    def __init__(self, name, kernel) -> None:
        self.name = name
        self.kernel = kernel
        self.children = []
        self.outputs = []
    
    def forward(self, state: StatePool):
        raise NotImplementedError
    
    def __call__(self, state: StatePool):
        result = self.forward(state)
        self.outputs.append(result)
        state.publish(result)

    def clean(self):
        self.outputs = []

    
class Workflow:
    def __init__(self) -> None:
        self.root: Module = None
        self.dest: Module = None
        self.modules: List[Module] = []
        self.edges: dict[Module, List[Module]] = defaultdict(list)
        self.states: StatePool = None
    
    def add_module(self, module: Module) -> None:
        self.modules.append(module)
    
    def add_edge(self, parent: Module, child: Module) -> None:
        parent.children.append(child)
        self.edges[parent].append(child)
    
    def set_root(self, module: Module) -> None:
        self.root = module
    
    def reset_modules(self) -> None:
        for module in self.modules:
            module.clean()
    
    def run(self,
            state,
            start_from: Optional[Module] = None,
            stop_at: Optional[Module] = None):
        
        if start_from is None:
            start_from = self.root
        
        q = queue.Queue()
        q.put(start_from)
        answer = None
        while not q.empty():
            module = q.get()
            module(state)
            answer = module.outputs[-1]
            for child in self.edges[module]:
                q.put(child)
            if module == stop_at:
                return module.outputs[-1]
        return answer
    
    def sort(self, predicate: callable) -> List[Module]:
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
        return [v for v in reversed(stack) if predicate(v)]