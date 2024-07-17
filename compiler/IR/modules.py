from dataclasses import dataclass, field
from compiler.IR.program import Module

@dataclass
class LMConfig:
    kwargs: dict = field(default_factory=dict)

class LLMPredictor(Module):
    def __init__(self, name, kernel) -> None:
        super().__init__(name=name, kernel=kernel)
        self.lm_history = []
        self.lm_config = None
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
