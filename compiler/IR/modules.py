from dataclasses import dataclass, field
from compiler.IR.program import Module
from abc import ABC, abstractmethod

class Input(Module):
    ...

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

    def forward(self, state):
        if self.lm is None:
            self.set_lm()
        result = self.kernel(state)
        self.lm_history.append(self.get_lm_history())
        return result
    
    
class CodeBox(Module):
    ...

@dataclass
class Retriever(Module):
        query_history: list = field(default_factory=list)
        retrieve_history: list = field(default_factory=list)
