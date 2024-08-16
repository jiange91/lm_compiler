from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple
import inspect
import time
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
    
    @property
    def all_news(self):
        report = {}
        for key in self.state:
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
        self.exec_times = []
        self.status = None
        self.input_fields, self.defaults = get_function_kwargs(kernel)
        logger.debug(f"Module {name} kernel has input fields {self.input_fields}")
    
    def forward(self, **kwargs):
        raise NotImplementedError
    
    def __call__(self, state: StatePool):
        kargs = {}
        for field in self.input_fields:
            if field not in self.defaults and field not in state.state:
                raise ValueError(f"Missing field {field} in state when calling {self.name}")
            if field in state.state:
                kargs[field] = state.news(field)
        # time the execution
        start = time.perf_counter()
        result = self.forward(**kargs)
        dur = time.perf_counter() - start
        self.outputs.append(result)
        self.exec_times.append(dur)
        state.publish(result)
        self.statis = ModuleStatus.SUCCESS

    def clean(self):
        self.outputs = []

class Input(Module):
    def __init__(self, input) -> None:
        name = '_user_input'
        self.input = input
        super().__init__(name=name, kernel=None)
    
    def forward(self, state: StatePool):
        return self.input

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
    ...

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