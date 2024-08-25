from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple
import inspect
import time
import logging
import copy
import threading

from compiler.IR.utils import get_function_kwargs

logger = logging.getLogger(__name__)

class State:
    def __init__(self, version_id, data, is_static) -> None:
        self.version_id = version_id
        self.data = data
        self.is_static = is_static # if True, the state will not be updated anymore

class StatePool:
    def __init__(self):
        self.states = defaultdict(list)
        
    def news(self, key: str, default = None):
        if key not in self.states or not self.states[key]:
            if default is None:
                raise ValueError(f"Key {key} not found in state")
            return default
        newest, version = None, -1
        for state in self.states[key]:
            if state.version_id > version:
                newest = state
                version = state.version_id
        return newest.data
    
    def init(self, kvs):
        self.publish(kvs, is_static = True, version_id = 0)
    
    def publish(self, kvs, is_static, version_id):
        for key, value in kvs.items():
            self.states[key].append(State(version_id, value, is_static))
    
    def all_news(self, fields = None):
        report = {}
        for key in self.states:
            if fields is not None and key not in fields:
                continue
            report[key] = self.news(key)
        return report
    
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
        self.outputs = []
        self.exec_times = []
        self.status = None
        self.is_static = False
        self.version_id = 0
        if kernel is not None:
            self.input_fields, self.defaults = get_function_kwargs(kernel)
        else:
            self.input_fields, self.defaults = None, None
        logger.debug(f"Module {name} kernel has input fields {self.input_fields}")
    
    def forward(self, **kwargs):
        raise NotImplementedError

    def __call__(self, statep: StatePool):
        if self.kernel is None:
            return
        for field in self.input_fields:
            if field not in self.defaults and field not in statep.states:
                raise ValueError(f"Missing field {field} in state when calling {self.name}")
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
        self.statis = ModuleStatus.SUCCESS
        self.version_id += 1

    def clean(self):
        self.outputs = []

class Input(Module):
    def __init__(self, name) -> None:
        super().__init__(name=name, kernel=None)

class Output(Module):
    def __init__(self, name) -> None:
        super().__init__(name=name, kernel=None)

@dataclass
class LMConfig(ABC):
    kwargs: dict = field(default_factory=dict)
    
    @abstractmethod
    def to_json(self):
        ...
    
    @abstractmethod
    def from_json(self, data):
        ...
 
class LLMPredictor(Module):
    def __init__(self, name, kernel) -> None:
        super().__init__(name=name, kernel=kernel)
        self.lm_history = []
        self.lm_config = None
        self.lm = None
    
    def clean(self):
        super().clean()
        self.lm_history = []
        self.lm = None
    
    def set_lm(self):
        raise NotImplementedError
    
    def get_lm_history(self):
        """
        Get token usage of each LLM call
        must include: {
            'prompt_tokens': int, 
            'completion_tokens': int,
            'model': str,
        }
        """
        raise NotImplementedError

    def forward(self, **kwargs):
        if self.lm is None:
            self.set_lm()
        result = self.kernel(**kwargs)
        self.lm_history.append(self.get_lm_history())
        return result
    
    
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

class Map(Module):
    ...