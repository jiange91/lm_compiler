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

from pydantic import BaseModel, Field
from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import get_buffer_string


@dataclass
class LMConfig(ABC):
    provider: str
    kwargs: dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'provider': self.provider,
            'kwargs': self.kwargs
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            provider=data['provider'],
            kwargs=data['kwargs']
        )
        
@dataclass
class Demonstration:
    # NOTE: current will try to convert all inputs to string
    # this might add long context if some input is a list of messages
    inputs: dict[str, str]
    
    # NOTE: currently use direct model output as reference output
    # this makes sense especially when the output should be structured
    output: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning: str = field(default=None)
    
    def to_string(self):
        """Naive string representation of the demonstration
        
        Semantic designner should have their own way to format the demonstration
        especially when the input contains other modalities
        """
        input_str = []
        for key, value in self.inputs.items():
            input_str.append(f"{key}:\n{value}")
        input_str = '**Input:**\n' + '\n'.join(input_str)
        if self.reasoning:
            input_str += f"\n**Reasoning:**\n{self.reasoning}"
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
    def get_output_schema(self) -> type[BaseModel] | None:
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
 
@dataclass
class TokenUsage:
    model: str
    prompt_tokens: int = field(default=0)
    completion_tokens: int = field(default=0)
    
    def __add__(self, other):
        if self.model != other.model:
            raise ValueError(f"Cannot add token usage of different models: {self.model} and {other.model}")
        return TokenUsage(
            model=self.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens
        )

    @staticmethod
    def pricing_pM(model, prompt, completion):
        if 'gpt-4o-mini' in model:
            return (0.15 * prompt +  0.6 * completion) / 1e6
        elif 'gpt-4o-2024-05-13' in model:
            return (5 * prompt + 15 * completion) / 1e6
        elif 'gpt-4o-2024-08-06' in model:
            return (2.5 * prompt + 10 * completion) / 1e6
        elif 'meta-llama/Llama-3.2-3B-Instruct-Turbo' in model:
            return 0.06 * (prompt + completion) / 1e6 # change to fireworks price
        elif 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo' in model:
            return 0.18 * (prompt + completion) / 1e6
        elif 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo' in model:
            return 0.18 * (prompt + completion) / 1e6
        elif 'meta-llama/Meta-Llama-3-8B-Instruct-Lite' in model:
            return 0.10 * (prompt + completion) / 1e6
        elif 'Qwen/Qwen2-72B-Instruct' in model:
            return 0.9 * (prompt + completion) / 1e6
        elif 'mistralai/Mistral-7B-Instruct-v0.3' in model:
            return 0.2 * (prompt + completion) / 1e6
        elif 'google/gemma-2-9b-it' in model:
            return 0.3 * (prompt + completion) / 1e6
        elif 'accounts/fireworks/models/llama-v3p2-3b-instruct' in model:
            return 0.1 * (prompt + completion) / 1e6
        else:
            raise ValueError(f"Model {model} pricing is not supported")
    
    def get_price(self):
        return self.pricing_pM(self.model, self.prompt_tokens, self.completion_tokens)

@dataclass
class TokenUsageSummary:
    total_price: float = field(default=0)
    token_consumed: dict[str, TokenUsage] = field(default_factory=dict)
    
    def __add__(self, other: 'TokenUsageSummary'):
        summary = copy.deepcopy(other)
        for model, usage in self.token_consumed.items():
            if model not in summary.token_consumed:
                summary.token_consumed[model] = copy.deepcopy(usage)
            else:
                summary.token_consumed[model] += usage
        return summary
    
    @classmethod
    def summarize(cls, records: Iterable[TokenUsage]) -> 'TokenUsageSummary':
        summary = cls()
        for record in records:
            if record.model not in summary.token_consumed:
                usage = TokenUsage(record.model, 0, 0)
                summary.token_consumed[record.model] = usage
            else:
                usage = summary.token_consumed[record.model]
            usage.prompt_tokens += record.prompt_tokens
            usage.completion_tokens += record.completion_tokens
            summary.total_price += record.get_price()
        return summary
    
    def __str__(self):
        return f"Total Price: {self.total_price} USD\n" + '\n'.join(
            [f"{model}: {usage.prompt_tokens} prompt tokens, {usage.completion_tokens} completion tokens" for model, usage in self.token_consumed.items()]
        )
    
class LLMPredictor(Module):
    def __init__(self, name, semantic: LMSemantic, lm, **kwargs) -> None:
        self.lm_history = []
        self.lm_config: LMConfig = None
        self.lm = lm
        self.input_cache = {}
        self.step_info = []
        self.rationale: str = None
        
        self.semantic = semantic
        # NOTE: lm and kernel will be set at first execution
        # this is to allow deepcopy of the module
        super().__init__(name=name, kernel=None, **kwargs)
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
        self.rationale = None
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

    def get_token_usage(self) -> list[TokenUsage]:
        """get current token usage of the LLM
        
        Please reset the usage cache at your will
        """
        records = []
        #NOTE: a LLMPredictor might have multiple LLMs in its history
        # if the config is dynamically changing
        for meta in self.lm_history:
            usage = TokenUsage(
                model=meta['model'],
                prompt_tokens=meta['prompt_tokens'],
                completion_tokens=meta['completion_tokens']
            )
            records.append(usage)
        return records

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
            'rationale': self.rationale,
            'output': lm_hist[-1]['response'],
        })
        self.rationale = None
        self.input_cache = {}
        self.lm_history.extend(lm_hist)
        
        return result
