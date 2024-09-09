from pydantic import BaseModel, Field
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Any

from compiler.IR.program import StatePool

@dataclass
class Input:
    type: type
    desc: str
    
class MetricMeta(ABCMeta):
    def __new__(cls, name, bases, attrs):
        required_inputs = set()
        for key, value in attrs.items():
            if isinstance(value, Input):
                required_inputs.add(key)
        attrs['required_inputs'] = required_inputs
        return super().__new__(cls, name, bases, attrs)
        

class MetricBase(ABC, metaclass=MetricMeta):
    
    def prepare_inputs(self, state: StatePool):
        inputs = {}
        for key in self.required_inputs:
            inputs[key] = state.news(key)
        return inputs    
    
    def __call__(self, label, state: StatePool):
        inputs = self.prepare_inputs(state)
        return self.score(label, **inputs)
    
    @abstractmethod
    def score(self, label, **inputs):
        pass
    
    
class ExactMatch(MetricBase):
    answer = Input(Any, "answer to the user question")
    
    def score(self, label, answer):
        return label == answer

if __name__ == '__main__':
    state = StatePool()
    state.init({'answer_1': 'dog'})
    
    metric = ExactMatch()
    print(metric('dog', state))