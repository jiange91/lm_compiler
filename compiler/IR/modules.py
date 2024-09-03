from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable, Union, Hashable
import inspect
import time
import logging
import copy
import threading
import concurrent.futures

from graphviz import Digraph

from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module, ComposibleModuleInterface, StatePool, Context

logger = logging.getLogger(__name__)

class Input(Module):
    def __init__(self, name) -> None:
        super().__init__(name=name, kernel=None)
    
    def forward(self, **kwargs):
        return {}

class Output(Module):
    def __init__(self, name) -> None:
        super().__init__(name=name, kernel=None)
    
    def forward(self, **kwargs):
        return {}

class Identity(Module):
    def __init__(self, name) -> None:
        super().__init__(name=name, kernel=None)
    
    def forward(self, **kwargs):
        return {}

    
class CodeBox(Module):
    def forward(self, **kwargs):
        return self.kernel(**kwargs)

class Retriever(Module):
    def __init__(self, name, kernel) -> None:
        super().__init__(name=name, kernel=kernel)
        self.query_history = []
        self.retrieve_history = []
    
    def clean(self):
        super().clean()
        self.query_history = []
        self.retrieve_history = []
    
    def forward(self, **kwargs):
        self.query_history.append(kwargs)
        result = self.kernel(**kwargs)
        self.retrieve_history.append(result)
        return result

class Map(Module, ComposibleModuleInterface):
    """Apply sub-graph to map over the input
    NOTE:
        sub_graph intermidiate states are not tracked within Map Module
        will make a deep copy of the input state for each sub-graph execution
    
    Args:
        kernel creates an iterable for the sub-graph
        make sure each item is independent as sub-graph execution is fully parallelized
    
    Examples:
        >>> sub_graph = CodeBox('whatever', lambda x, y: x + y) # any kinds of basic blocks
        >>> def map_kernel(xs: list[x], y):
        >>>     for x in xs:
        >>>         yield {'x': x, 'y': y}
        >>> map_module = Map('map', map_kernel, sub_graph)
    """
    def __init__(
        self, 
        name, 
        sub_graph: Module, 
        map_kernel: Callable, 
        output_fields: Union[str, list[str]], 
        max_parallel: int = 5
    ) -> None:
        super().__init__(name, map_kernel)
        self.sub_graph = sub_graph
        self.map_kernel = map_kernel
        self.output_fields = output_fields if isinstance(output_fields, list) else [output_fields]
        self.max_parallel = max_parallel
    
    def forward(self, **kwargs):
        tracked_states = []
        def new_input_gen():
            for item in self.map_kernel(**kwargs):
                state_input = StatePool()
                state_input.init(copy.deepcopy(item))
                tracked_states.append(state_input)
                yield state_input
                
        results = {field: [] for field in self.output_fields}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            executor.map(self.sub_graph, new_input_gen())
        for statep in tracked_states:
            for field in self.output_fields:
                results[field].append(statep.news(field))
        return results

    def immediate_submodules(self) -> List[Module]:
        return [self.sub_graph]

    def reset(self):
        super().reset()
        self.sub_graph.reset()
    
    def replace_node_handler(self, old_node: Module, new_node_in: Module, new_node_out: Module) -> bool:
        assert new_node_in is new_node_out, "Map Module only accept a single-node replacement"
        if old_node is not self.sub_graph:
            logger.warning(f"Node {old_node.name} not found in {self.name}")
            return False
        self.sub_graph = new_node_in
        return True

    def _visualize(self, dot: Digraph):
        dot.node(f'_{self.name}_cluster_ancor', style='invis', fixedsize='true', width='0', height='0')
        if isinstance(self.sub_graph, ComposibleModuleInterface):
            with dot.subgraph(name=f'cluster_{self.sub_graph.name}') as s:
                self.sub_graph._visualize(s)
        else:
            dot.node(self.sub_graph.name)

class Branch(Module):
    def __init__(
        self,
        name: str,
        src: list[str],
        multiplexier: Callable[..., Union[Hashable, list[Hashable]]],
        destinations: list[str],
        multiplexier_str: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, kernel=multiplexier)
        self.src = src
        self.multiplexier = multiplexier
        self.destinations = destinations
        self.invoke_times = 0
        
        if multiplexier_str is not None:
            self.multiplexier_str = multiplexier_str
        else:
            self.multiplexier_str = inspect.getsource(multiplexier)
        
    def on_signature_generation(self):
        try:
            self.input_fields.remove('ctx')
        except ValueError:
            pass
        self.defaults.pop('ctx', None)
    
    def forward(self, **kwargs):
        ctx = Context(self, self.name, self.invoke_times)
        dest = self.kernel(ctx, **kwargs)
        dest = dest if isinstance(dest, list) else [dest]
        self.invoke_times += 1
        return {self.name + '#branch_result': dest}