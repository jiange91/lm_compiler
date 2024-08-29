from compiler.IR.base import Module
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable, Union
from langchain_core.pydantic_v1 import BaseModel, Field
import inspect
import time
import logging
import copy
import threading
import concurrent.futures

from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module, ComposibleModuleInterface, StatePool

@dataclass
class LMConfig(ABC):
    kwargs: dict = field(default_factory=dict)
    
    @abstractmethod
    def to_json(self):
        ...
    
    @abstractmethod
    def from_json(self, data):
        ...

class LMSemantic(ABC):
    
    @abstractmethod
    def get_agent_role(self) -> str:
        ...
        
    @abstractmethod
    def get_agent_inputs(self) -> list[str]:
        ...
        
    @abstractmethod
    def get_agent_outputs(self) -> list[str]:
        ...
    
    @abstractmethod
    def get_invoke_routine(self) -> Callable:
        ...
    
    @abstractmethod
    def get_formatted_info(self) -> str:
        ...
    
 
class LLMPredictor(Module):
    def __init__(self, name, semantic: LMSemantic) -> None:
        super().__init__(name=name, kernel=semantic.get_invoke_routine())
        self.lm_history = []
        self.lm_config = None
        self.lm = None
        self.semantic = semantic

    def on_signature_generation(self):
        try:
            self.input_fields.remove('lm')
        except ValueError:
            pass
        try:
            self.input_fields.remove('llm')
        except ValueError:
            pass
        self.defaults.pop('lm', None)
        self.defaults.pop('llm', None)
    
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
        return type must be a list of dict
        """
        raise NotImplementedError

    def forward(self, **kwargs):
        if self.lm is None:
            self.set_lm()
        result = self.kernel(self.lm, **kwargs)
        self.lm_history.extend(self.get_lm_history())
        return result