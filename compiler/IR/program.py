from typing import List, Optional, Tuple
from collections import defaultdict
from graphviz import Digraph
from compiler.IR.modules import Module, StatePool, ModuleStatus, LLMPredictor
import json

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

    
class Workflow:
    def __init__(self) -> None:
        self.modules: List[Module] = []
        self.edges: dict[Module, List[Module]] = defaultdict(list)
        self.states: StatePool = None
        self.exit_point: Tuple[Module, str] = None
        self.dot = Digraph()
        self.token_usage_buffer = {'total': {}}
    
    def add_module(self, module: Module) -> None:
        self.modules.append(module)
        self.dot.node(module.name)
    
    def add_edge(self, parent: Module, child: Module) -> None:
        parent.children.append(child)
        child.dependencies.append(parent)
        self.edges[parent].append(child)
        self.dot.edge(parent.name, child.name)
    
    def set_exit_point(self, module: Module, field: str) -> None:
        self.exit_point = (module, field)
    
    def reset_modules(self, clear_token_buffer = False) -> None:
        self.update_token_usage_summary()
        if clear_token_buffer:
            self.token_usage_buffer = {'total': {}}
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
            logger.info(f"Running module {module.name}")
            if start_from is None or (module is start_from and not started):
                started = True
            if not started:
                module.status = ModuleStatus.SKIPPED
                logger.info(f"Skipping module {module.name}")
                continue
            if module == stop_before:
                logger.info(f"Stopping before module {module.name}")
                return answer
            deps = module.dependencies
            if deps is None or all(m.statis is ModuleStatus.SUCCESS for m in deps):
                module(state)
                answer = module.outputs[-1]
            else:
                module.statis = ModuleStatus.FAILED
                raise ValueError(f"Module {module.name} failed to run due to dependencies")
            if module == self.exit_point[0]:
                return self.exit_result
        return self.exit_result
    
    @property
    def exit_result(self):
        if self.exit_point is None:
            raise ValueError("No exit point set")
        return {self.exit_point[1]: self.exit_point[0].outputs[-1][self.exit_point[1]]} 
    
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

    def visualize(self, fpath):
        self.dot.render(directory=fpath)
    
    def update_token_usage_summary(self):
        for lm in (m for m in self.modules if isinstance(m, LLMPredictor)):
            if lm.name not in self.token_usage_buffer:
                self.token_usage_buffer[lm.name] = {}
            for meta in lm.lm_history:
                model = meta['model']
                if model not in self.token_usage_buffer[lm.name]:
                    self.token_usage_buffer[lm.name][model] = defaultdict(int)
                self.token_usage_buffer[lm.name][model]['prompt_tokens'] += meta['prompt_tokens']
                self.token_usage_buffer[lm.name][model]['completion_tokens'] += meta['completion_tokens']
                if model not in self.token_usage_buffer['total']:
                    self.token_usage_buffer['total'][model] = defaultdict(int)
                self.token_usage_buffer['total'][model]['prompt_tokens'] += meta['prompt_tokens']
                self.token_usage_buffer['total'][model]['completion_tokens'] += meta['completion_tokens']
            # NOTE: clear incase of double counting
            lm.lm_history = []
    
    def log_module_time(self, path):
        import numpy as np
        times = {}
        for module in self.modules:
            times[module.name] = np.mean(module.exec_times)
        with open(path, 'w+') as f:
            json.dump(times, f, indent=4)
        
    
    def log_token_usage(self, path):
        self.update_token_usage_summary()
        with open(path, 'w+') as f:
            json.dump(self.token_usage_buffer, f, indent=4)