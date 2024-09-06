from typing import List, Optional, Tuple, Callable, Hashable, Union, Any
from collections import defaultdict, deque
from graphviz import Digraph
from functools import partial

from compiler.IR.llm import LLMPredictor
from compiler.IR.modules import Input, Output, Branch, Identity
from compiler.IR.base import *
from compiler.IR.rewriter.utils import replace_branch_return_destination
from compiler.IR.utils import get_function_kwargs, simple_cycles
from dataclasses import dataclass
import json
from functools import wraps
import concurrent.futures
import time
from itertools import chain
        

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


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
    
class Workflow(ComposibleModuleInterface):
    dependencies = tuple[str, ...]
    
    def __init__(self, name) -> None:
        self.modules: dict[str, Module] = {}
        self.static_dependencies: dict[Workflow.dependencies, list[str]] = defaultdict(list)
        self.branches: dict[str, Branch] = {}
        
        """
        Following will be (re)populated during (re)compilation
        """
        self.start: Input = None
        self.end: Output = None
        # edges are not used for preparing the next module to run
        self.edges: dict[str, list[str]] = defaultdict(list)
        # the previous module will notify the trigger when it's done
        self.triggers: list[Trigger] = []
        self.publish_channels : dict[str, list[Trigger]] = defaultdict(list)
        
        """
        some runtime meta
        """
        self.token_usage_buffer = {'total': {}}
        self.current_step = 0
        super().__init__(name, None)
    
    def add_module(self, module: Module, reset_parent = False) -> None:
        self.sub_module_validation(module, reset_parent)
        self.modules[module.name] = module
        if reset_parent:
            module.enclosing_module = self
    
    def _edge_validation(self, src: list[str], dest: list[str]):
        for s in src:
            if s not in self.modules:
                raise ValueError(f"Source module {s} not found, please add the module first")
        for d in dest:
            if d not in self.modules:
                raise ValueError(f"Destination module {d} not found, please add the module first")
    
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
            src = sorted(list(set([src])))
        if isinstance(dest, str):
            dest = sorted(list(set([dest])))
        self._edge_validation(src, dest)
        for s in src:
            self.modules[s].enclosing_module = self
        for d in dest:
            self.modules[d].enclosing_module = self
        self.static_dependencies[tuple(src)].extend(dest)
                
    def add_branch(
        self, 
        name: str,
        src: Union[str, list[str]],
        multiplexier: Callable[..., Union[Hashable, list[Hashable]]],
        multiplexier_str: Optional[str] = None,
    ) -> None:
        """add control flow
        
        If need complex synchronization, use add_edge to add a preceding module
        
        Args:
            src (Union[str, list[str]]):
                the module that the control flow need to synchronize with
            
            multiplexier (callable): 
                signature should be (ctx, arg1, arg2, ...) -> Hashable
                ctx contains some useful information for the multiplexier to make decision
                
        Examples:
        NOTE: please hint all possible destinations for the multiplexier
            ```python
            from compiler.IR.program import hint_possible_destinations
            
            @hint_possible_destinations(['a', 'b'])
            def multiplexier(ctx, smth):
            ... if f(smth):
            ...    return ['a', 'b]
            ... else:
            ...    return 'b'
            ```
        """
        src = sorted(list(set([src])))
        self._edge_validation(src, multiplexier._possible_destinations)
        
        branch = Branch(name, src, multiplexier, multiplexier._possible_destinations, multiplexier_str)
        self.add_module(branch)
        self.branches[name] = branch
        self.add_edge(src, branch.name)
    
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
        self.start = None
        self.end = None
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
        for srcs, dests in self.static_dependencies.items():
            immediate_deps = set(srcs)
            potential_deps = set().union(chain.from_iterable(dep_graph[src] for src in srcs))
            def make_foo(dests):
                return lambda statep: dests
            trigger = Trigger(immediate_deps, potential_deps, make_foo(dests))
            self.triggers.append(trigger)
            for src in srcs:
                self.publish_channels[src].append(trigger)
        # same for dynamic branches
        for src, branch in self.branches.items():
            immediate_deps = {src}
            potential_deps = dep_graph[src]
            def make_dfoo(src, statep):
                return statep.news(src + '#branch_result')
            trigger = Trigger(immediate_deps, potential_deps, partial(make_dfoo, src))
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
    
    
    def reset(self) -> None:
        # clear sub-llms history
        self.update_token_usage_summary()
        self.token_usage_buffer = {'total': {}}
            
        for trigger in self.triggers:
            trigger.active = False
            trigger.notified.clear()
        self.current_step = 0
               
        for module in self.immediate_submodules():
            module.reset()
    
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
    
    def visualize(self, dir: str):
        dot = Digraph()
        dot.attr(compound='true')
        self._visualize(dot)
        dot.render(directory=dir)

    def _visualize(self, dot: Digraph):
        dot.node(f'_{self.name}_cluster_ancor', style='invis', fixedsize='true', width='0', height='0')
        for srcs, dests in self.static_dependencies.items():
            for src in srcs:
                attrs = {}
                if isinstance(self.modules[src], ComposibleModuleInterface):
                    attrs['ltail'] = f'cluster_{src}'
                    src = f'_{src}_cluster_ancor'
                for dest in dests:
                    cattrs = {**attrs}
                    if isinstance(self.modules[dest], ComposibleModuleInterface):
                        cattrs['lhead'] = f'cluster_{dest}'
                        dest = f'_{dest}_cluster_ancor'
                    dot.edge(src, dest, **cattrs)
        for name, branch in self.branches.items():
            attrs = {}
            for dest in branch.destinations:
                cattrs = {**attrs}
                if isinstance(self.modules[dest], ComposibleModuleInterface):
                    cattrs['lhead'] = f'cluster_{dest}'
                    dest = f'_{dest}_cluster_ancor'
                dot.edge(name, dest, style='dashed', **cattrs)
        for name, m in self.modules.items():
            if isinstance(m, ComposibleModuleInterface):
                with dot.subgraph(name=f'cluster_{name}') as s:
                    m._visualize(s)
                    s.attr(label=name)
    
    def update_token_usage_summary(self):
        """get token usage summary for all LLM modules recursively
        """
        for lm in (m for m in self.get_all_modules(lambda x: isinstance(x, LLMPredictor))):
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
    
    def immediate_submodules(self) -> List[Module]:
        return list(self.modules.values())

    def get_all_modules(self, predicate):
        """get all modules that satisfy the predicate
        
        will search recursively into all composible modules
        """
        module_queue = deque(self.immediate_submodules())
        result = []
        
        while module_queue:
            module = module_queue.popleft()
            if predicate(module):
                result.append(module)
            if isinstance(module, ComposibleModuleInterface):
                module_queue.extend(module.immediate_submodules())
        return result

    def bypass_node(self, node_name):
        node_in_deps:list[Union[Branch, Tuple[str, ...]]] = []
        node_in_dests: list[Union[Branch, Tuple[str, ...]]] = []
        for deps, dests in self.static_dependencies.items():
            if node_name in deps:
                node_in_deps.append(deps)
            if node_name in dests:
                node_in_dests.append(deps)
        for branch in self.branches.values():
            if node_name in branch.src:
                node_in_deps.append(branch)
            if node_name in branch.destinations:
                node_in_dests.append(branch)
                
        if node_in_deps:
            assert len(node_in_deps) > 0, f"Node {node_name} is in dependency but no one activates it"
        
        counter = 0
        new_dynamic_dests = []
        for deps_to_replace in node_in_deps:
            for replace_candidate in node_in_dests:
                if isinstance(deps_to_replace, Branch):
                    if isinstance(replace_candidate, Branch):
                        buffer_id = f'sync_buffer_{replace_candidate.name}_to_{deps_to_replace.name}'
                        self.add_module(Identity(buffer_id))
                        
                        deps_to_replace.src[deps_to_replace.src.index(node_name)] = buffer_id
                        new_dynamic_dests.append(replace_candidate.name)
                    else:
                        pass
                else:
                    pass

    def replace_node_handler(self, old_node: Module, new_node_in: Module, new_node_out: Module) -> bool:
        if isinstance(old_node, Branch):
            NotImplementedError("Branch replacement is not supported yet")
            
        if old_node.name not in self.modules:
            return False
        del self.modules[old_node.name]
        
        # replace in static dependencies
        sync_barriers = list(self.static_dependencies.keys())
        for sb in sync_barriers:
            # check this first in case cycle of one node
            if old_node.name in (dests := self.static_dependencies[sb]):
                dests[dests.index(old_node.name)] = new_node_in.name
            # then update the key
            if old_node.name in sb:
                new_sb = list(sb)
                new_sb[sb.index(old_node.name)] = new_node_out.name
                self.static_dependencies[tuple(new_sb)] = self.static_dependencies.pop(sb)
        
        # replace branches
        for name, branch in self.branches.items():
            if old_node.name in (dests := branch.destinations):
                dests[dests.index(old_node.name)] = new_node_in.name # this also updates multiplexier's hint
                # also replace function return
                new_multiplexier, new_code_str = replace_branch_return_destination(branch.multiplexier, old_node.name, new_node_in.name, branch.multiplexier_str)
                branch.multiplexier = new_multiplexier
                branch.multiplexier_str = new_code_str
            if old_node.name in branch.src:
                branch.src[branch.src.index(old_node.name)] = new_node_out.name
        return True