from compiler.IR.base import Module
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

from langchain_core.pydantic_v1 import BaseModel, Field
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
    def get_formatted_info(self) -> str:
        ...
        
    @abstractmethod
    def get_high_level_info(self) -> str:
        ...
    
    @abstractmethod
    def get_output_schema(self) -> BaseModel:
        ...
    
 
class LLMPredictor(Module):
    def __init__(self, name, semantic: LMSemantic, lm) -> None:
        self.lm_history = []
        self.lm_config = {}
        self.lm = lm
        
        self.semantic = semantic
        super().__init__(name=name, kernel=self.get_invoke_routine())
        
    @property
    def lm(self):
        return self._lm
    
    @lm.setter
    def lm(self, value):
        self._lm = value
    
    def get_invoke_routine(self):
        raise NotImplementedError

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
    
    def reset(self):
        super().reset()
        self.lm_history = []
        self.lm = None
        self.custom_reset()
    
    def custom_reset(self):
        raise NotImplementedError
    
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
        result = self.kernel(**kwargs)
        self.lm_history.extend(self.get_lm_history())
        return result