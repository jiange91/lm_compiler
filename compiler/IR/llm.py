from compiler.IR.base import Module
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable, Union, Any
import inspect
import time
import logging
import copy
import concurrent.futures
import uuid

from langchain_core.pydantic_v1 import BaseModel, Field
from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module, ComposibleModuleInterface, StatePool
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import get_buffer_string


@dataclass
class LMConfig(ABC):
    kwargs: dict = field(default_factory=dict)
    
    @abstractmethod
    def to_json(self):
        ...
    
    @abstractmethod
    def from_json(self, data):
        ...
        
@dataclass
class Demonstration:
    # NOTE: current will try to convert all inputs to string
    # this might add long context if some input is a list of messages
    inputs: dict[str, str]
    
    # NOTE: currently use direct model output as reference output
    # this makes sense especially when the output should be structured
    output: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_string(self):
        """Naive string representation of the demonstration
        
        Semantic designner should have their own way to format the demonstration
        especially when the input contains other modalities
        """
        input_str = []
        for key, value in self.inputs.items():
            input_str.append(f"{key}:\n{value}")
        input_str = '**Input:**\n' + '\n'.join(input_str)
        
        demo_str = f"{input_str}\n**Answer:**\n{self.output}"
        return demo_str
    

class LMSemantic(ABC):
    
    @abstractmethod
    def prompt_fully_manageable(self) -> bool:
        """If the semantic can be fully managed by the compiler
        
        Currently, all semantic that have static prompt template can be fully managed
        i.e. the following_messages is not specified
        """
        ...
    
    @abstractmethod
    def get_agent_role(self) -> str:
        ...
        
    @abstractmethod
    def get_agent_inputs(self) -> list[str]:
        ...
    
    @abstractmethod
    def get_img_input_names(self) -> list[str]:
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
    
    @abstractmethod
    def get_output_spec(self) -> Tuple[bool, Optional[str]]:
        ...
        
    @abstractmethod
    def get_demos(self) -> list[Demonstration]:
        ...
    
    @abstractmethod
    def set_demos(self, demos: list[Demonstration]):
        ...
    
 
class LLMPredictor(Module):
    def __init__(self, name, semantic: LMSemantic, lm) -> None:
        self.lm_history = []
        self.lm_config = {}
        self.lm = lm
        self.input_cache = {}
        self.step_info = []
        
        self.semantic = semantic
        # NOTE: lm and kernel will be set at first execution
        # this is to allow deepcopy of the module
        super().__init__(name=name, kernel=None)
        self.input_fields = self.semantic.get_agent_inputs()
        
    @property
    def lm(self):
        return self._lm
    
    @lm.setter
    def lm(self, value):
        self._lm = value
    
    def initialize_kernel(self):
        self.kernel = self.get_invoke_routine()
        self.prepare_input_env()
    
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
        self.step_info = []
        self.input_cache = {}
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
            'response': str,
            'model': str,
        }
        return type must be a list of dict
        """
        raise NotImplementedError
    
    def get_step_as_example(self) -> Demonstration:
        """Get invocation info of this LLM
       
        A LLM can be called multiple times in one workflow invocation
        currently this function will only return the last step to be used for bootstrapping few-shot examples
        """
        raise NotImplementedError

    def on_invoke(self, kwargs: dict):
        self.input_cache = kwargs
    
    def forward(self, **kwargs):
        if self.lm is None:
            # if lm is reset or not set, initialize it and the kernel
            self.set_lm()
            self.initialize_kernel()
            
        if self.kernel is None:
            self.initialize_kernel()
            
        result = self.kernel(**kwargs)
        
        lm_hist = self.get_lm_history()
        self.step_info.append({
            'inputs': copy.deepcopy(self.input_cache),
            'output': lm_hist[-1]['response'],
        })
        self.input_cache = {}
        self.lm_history.extend(lm_hist)
        
        return result