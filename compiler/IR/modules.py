from dataclasses import dataclass, field
from compiler.IR.program import Module, StatePool
from abc import ABC, abstractmethod

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