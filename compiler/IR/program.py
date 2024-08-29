from typing import List, Optional, Tuple, Callable, Hashable, Union, Any
from collections import defaultdict, deque
from graphviz import Digraph
from compiler.IR.llm import LLMPredictor
from compiler.IR.modules import Input, Output
from compiler.IR.base import Module, StatePool, ModuleStatus, ComposibleModuleInterface
from compiler.IR.utils import get_function_kwargs, simple_cycles
from dataclasses import dataclass
import json
from functools import wraps
import concurrent.futures
import time
from itertools import chain
        

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class Context:
    statep: StatePool
    predecessor: str
    invoke_time: int # start from 1

def hint_possible_destinations(dests: list[str]):
    def hinter(func):
        func._possible_destinations = dests
        return func
    return hinter

class Branch:
    """
    multiplexier signature should be (ctx, **kwargs) -> Hashable
    
    ctx contains some useful information for the multiplexier to make decision
    """
    def __init__(
        self,
        src: Module,
        multiplexier: Callable[..., Union[Hashable, list[Hashable]]],
        destinations: list[str],
    ):
        self.src = src
        self.multiplexier = multiplexier
        self.destinations = destinations
    
        input_fields, defaults = get_function_kwargs(self.multiplexier)
        try:
            input_fields.remove('ctx')
        except ValueError:
            pass
        defaults.pop('ctx', None)
        
        def new_multiplexier(statep: StatePool):
            for field in input_fields:
                if field not in statep.states and field not in defaults:
                    raise ValueError(f"Field {field} not found in state")
            ctx = Context(statep, self.src.name, self.src.version_id)
            kargs = {field: statep.news(field) for field in statep.states if field in input_fields}
            dests = self.multiplexier(ctx, **kargs)
            if isinstance(dests, str):
                dests = [dests]
            return dests
        self.exposed_multiplexier = new_multiplexier

class Trigger:
    """synchronization barrier
    This decides when synchronous barrier is satisfied
    thus decide what next module can be invoked
    """
    def __init__(
        self, 
        immediate_deps: set[str], 
        potential_deps: set[str], 
        prepare_next_steps: Callable[[StatePool], list[str]],
    ):
        self.immediate_deps = immediate_deps
        self.potential_deps = potential_deps
        self.prepare_next_steps = prepare_next_steps
        self.notified: dict[str, bool] = {}
        self.active = False
    
    def notify(self, src: str, is_static: bool):
        if src in self.notified:
            raise ValueError(f"Trigger {self} has already been notified by {src} and not consumed yet")
        self.active = True
        self.notified[src] = is_static
    
    def can_fire(self, scheduled_modules: set[str]) -> set[str]:
        """determine if this trigger can invoke the module
        
        1. If all immediate deps are notified, fire
        2. If any potential deps are scheduled, wait, otherwise fire
        """
        if self.immediate_deps == set(self.notified.keys()):
            self.active = False
            return set(self.notified.keys())
        if not self.potential_deps.intersection(scheduled_modules):
            self.active = False
            return set(self.notified.keys())
        return None
    
    def consume(self, notified: set[str]):
        for notif in notified:
            if notif in self.notified and not self.notified[notif]:
                del self.notified[notif]
    
class Workflow(Module, ComposibleModuleInterface):
    dependencies = tuple[str, ...]
    
    def __init__(self, name) -> None:
        self.start: Input = None
        self.end: Output = None
        self.modules: dict[str, Module] = {}
        # NOTE: edges are not used for preparing the next module to run
        self.edges: dict[str, list[str]] = defaultdict(list)
        self.static_dependencies: dict[Workflow.dependencies, list[str]] = defaultdict(list)
        self.branches: dict[str, Branch] = {}
        
        # the previous module will notify the trigger when it's done
        self.triggers: list[Trigger] = []
        self.publish_channels : dict[str, list[Trigger]] = defaultdict(list)
        
        self.dot = Digraph()
        self.token_usage_buffer = {'total': {}}
        self.current_step = 0
        super().__init__(name, None)
    
    def add_module(self, module: Module) -> None:
        self.modules[module.name] = module
        self.dot.node(module.name)
    
    def add_edge(self, src: Union[str, list[str]], dest: Union[str, list[str]]) -> None:
        """add static dataflow
        
        Args:
            src (Union[str, list[str]]): source module(s)
            dest (Union[str, list[str]]): destination module(s)
            
        NOTE:
            src added in one add_edge call will be treated as a synchronization group
            i.e. the dest module will only run after all src modules are done
            
            If you prefer individual triggers for each src module, call add_edge multiple times
        """
        if isinstance(src, str):
            src = [src]
        if isinstance(dest, str):
            dest = [dest]
        self.static_dependencies[tuple(src)].extend(dest)
        for s in src:
            for d in dest:
                self.dot.edge(s, d)
        
    def add_branch(
        self, 
        src: str,
        multiplexier: Callable[..., Union[Hashable, list[Hashable]]]) -> None:
        """add control flow
        
        If need complex synchronization, use add_edge to add a preceding module
        
        Args:
            src (str): 
                the module that the control flow starts from
            
            multiplexier (callable): 
                signature should be (ctx, **kwargs) -> Hashable
                ctx contains some useful information for the multiplexier to make decision
            
            child_map (dict):
                a mapping from multiplexier's return value to next modules
                
        Examples:
        NOTE: please hint all possible destinations for the multiplexier
            ```python
            from compiler.IR.program import hint_possible_destinations
            
            @hint_possible_destinations(['a', 'b'])
            def multiplexier(ctx, **kwargs):
            ... if smth:
            ...    return ['a', 'b]
            ... else:
            ...    return 'b'
            ```
        """
        branch = Branch(self.modules[src], multiplexier, multiplexier._possible_destinations)
        self.branches[src] = branch
        for dest in branch.destinations:
            self.dot.edge(src, dest, style='dashed')
            
    
    def get_dependency(self):
        # build hyperthetical reversed graph
        re_edges: dict[str, list[str]] = defaultdict(list)
        for srcs, dests in self.static_dependencies.items():
            for dest in dests:
                re_edges[dest].extend(srcs)
        for src, branch in self.branches.items():
            for dest in branch.destinations:
                re_edges[dest].append(src)
        # remove duplication
        for dest, src in re_edges.items():
            re_edges[dest] = list(set(src))
        
        # get all dependent nodes given destination node
        def dfs(dest, visited: set[str]):
            if dest in visited:
                return visited
            visited.add(dest)
            reachable = {dest}
            for src in re_edges[dest]:
                reachable.update(dfs(src, visited))
            visited.remove(dest)
            return reachable
        
        dependency_graph: dict[str, set[str]] = {}
        for module in self.modules:
            dependency_graph[module] = dfs(module, set())
        return dependency_graph
    
    def validate(self):
        # Clear previous compilation
        self.edges.clear()
        self.triggers.clear()
        self.publish_channels.clear()
        
        # TODO: add more validation check
        for name, module in self.modules.items():
            if isinstance(module, Input):
                if self.start is not None:
                    raise ValueError("Multiple start points detected")
                self.start = module
            if isinstance(module, Output):
                if self.end is not None:
                    raise ValueError("Multiple end points detected")
                self.end = module
        if self.start is None:
            raise ValueError("No start point detected")
        if self.end is None:
            raise ValueError("No end point detected")

    def compile(self):
        """config each module with graph analysis
        """
        self.validate()
        # Compile all subgraphs
        all_sub_graphs = self.get_all_modules(lambda x: isinstance(x, Workflow))
        for sub_graph in all_sub_graphs:
            sub_graph.compile()
        # Derive the dependency graph
        dep_graph: dict[str, set[str]] = self.get_dependency()
        # For each module, register their triggers and corresponding dependencies
        def make_foo(dests):
            return lambda statep: dests
        for srcs, dests in self.static_dependencies.items():
            immediate_deps = set(srcs)
            potential_deps = set().union(chain.from_iterable(dep_graph[src] for src in srcs))
            trigger = Trigger(immediate_deps, potential_deps, make_foo(dests))
            self.triggers.append(trigger)
            for src in srcs:
                self.publish_channels[src].append(trigger)
        # same for dynamic branches
        for src, branch in self.branches.items():
            immediate_deps = {src}
            potential_deps = dep_graph[src]
            trigger = Trigger(immediate_deps, potential_deps, branch.exposed_multiplexier)
            self.triggers.append(trigger)
            self.publish_channels[src].append(trigger)
        # Identify dynamic modules, i.e. steps will be invoked multiple times
        for srcs, dests in self.static_dependencies.items():
            for src in srcs:
                self.edges[src].extend(dests)
        for src, branch in self.branches.items():
            self.edges[src].extend(branch.destinations)
        # remove duplication
        for src, dests in self.edges.items():
            self.edges[src] = list(set(dests))
        for name in self.modules:
            if name not in self.edges:
                self.edges[name] = []
        nodes_in_cycles = set(chain.from_iterable(simple_cycles(self.edges)))
        for name in nodes_in_cycles:
            self.modules[name].is_static = False
        # TODO: populate history states
    
    def reset_modules(self, clear_token_buffer = False) -> None:
        self.update_token_usage_summary()
        if clear_token_buffer:
            self.token_usage_buffer = {'total': {}}
            
        for trigger in self.triggers:
            trigger.active = False
            trigger.notified.clear()
               
        for module in self.modules.values():
            module.clean()
        self.current_step = 0
    
    def exec_module(self, module: Module, statep: StatePool):
        try:
            module(statep)
        except Exception as e:
            logger.error(f"Error in {module.name}: {e}")
            module.status = ModuleStatus.FAILED
            raise e
            
        if module.status == ModuleStatus.SUCCESS:
            if module.name in self.publish_channels:
                for next_to_notify in self.publish_channels[module.name]:
                    next_to_notify.notify(module.name, module.is_static)
    
    def fire_next_round(self, statep: StatePool, scheduled: set[str] = None, stop_before: str = None):
        candidates = set()
        triggered_notifs = set()
        fired_triggers = []
        # check immediate deps
        for trigger in self.triggers:
            if trigger.active and (notifs := trigger.can_fire(scheduled)):
                candidates.update(trigger.prepare_next_steps(statep))
                triggered_notifs.update(notifs)
                fired_triggers.append(trigger)
        # after scheduling, check potential deps
        for trigger in self.triggers:
            if trigger.active and (notifs := trigger.can_fire(candidates)):
                candidates.update(trigger.prepare_next_steps(statep))
                triggered_notifs.update(notifs)
                fired_triggers.append(trigger)
        # invalidate triggered notifications
        for trigger in fired_triggers:
            trigger.consume(triggered_notifs)
        # pure stop_before from candidates
        if stop_before is not None:
            candidates.discard(stop_before)
        return candidates

    def pregel_run(
        self,
        statep,
        start_from: Optional[str] = None,
        stop_before: Optional[str] = None,
    ):
        scheduled = set()
        if start_from is None:
            start_from = self.start.name
        scheduled.add(start_from)
        
        while True:
            num_tasks = len(scheduled)
            if num_tasks == 0:
                break
            logger.debug(f"Step {self.current_step}: {scheduled}")
            # with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
            #     futures = [executor.submit(self.exec_module, self.modules[name], statep) for name in scheduled]
            #     concurrent.futures.wait(futuresa)
            for name in scheduled:
                self.exec_module(self.modules[name], statep)
            scheduled = self.fire_next_round(statep, scheduled, stop_before)
            self.current_step += 1
    
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
            for child in self.static_dependencies[v]:
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
        for lm in (m for m in self.modules.values() if isinstance(m, LLMPredictor)):
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
        for module in self.modules.values():
            times[module.name] = np.mean(module.exec_times)
        with open(path, 'w+') as f:
            json.dump(times, f, indent=4)
        
    
    def log_token_usage(self, path):
        self.update_token_usage_summary()
        with open(path, 'w+') as f:
            json.dump(self.token_usage_buffer, f, indent=4)
    
    def __call__(self, statep: StatePool):
        start = time.perf_counter()
        self.pregel_run(statep)
        dur = time.perf_counter() - start
        # update metadata
        self.exec_times.append(dur)
        self.status = ModuleStatus.SUCCESS
        self.version_id += 1
    
    def clean(self):
        self.reset_modules()
    
    def immediate_submodules(self) -> List[Module]:
        return list(self.modules.values())

    def get_all_modules(self, predicate):
        module_queue = deque(self.immediate_submodules())
        result = []
        
        while module_queue:
            module = module_queue.popleft()
            if predicate(module):
                result.append(module)
            if isinstance(module, ComposibleModuleInterface):
                module_queue.extend(module.immediate_submodules())
        return result

    def replace_node_handler(self, old_node: Module, new_node: Module) -> bool:
        pass