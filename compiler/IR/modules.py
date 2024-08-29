from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable, Union
import inspect
import time
import logging
import copy
import threading
import concurrent.futures

from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module, ComposibleModuleInterface, StatePool

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
        >>> sub_graph = CodeBox('whatever', lambda x, y: x + y)
        >>> def map_kernel(xs: list[x], y):
        >>>     for x in xs:
        >>>         yield {'x': x, 'y': y}
        >>> map_module = Map('map', map_kernel, sub_graph)
    """
    def __init__(
        self, 
        name, 
        sub_graph, 
        map_kernel, 
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
    
    def replace_node_handler(self, old_node: Module, new_node: Module) -> bool:
        pass