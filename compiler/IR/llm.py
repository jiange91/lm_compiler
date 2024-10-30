from compiler.IR.base import Module
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, auto
from typing import List, Optional, Tuple, Iterable, Callable, Union, Any, Literal
import threading
import inspect
import time
import logging
import copy
import concurrent.futures
import json
import uuid

from pydantic import BaseModel, Field
from compiler.IR.utils import get_function_kwargs
from compiler.IR.base import Module
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.utils import get_buffer_string

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    prompt_tokens: int = field(default=0)
    completion_tokens: int = field(default=0)
    prompt_cached_tokens: int = field(default=0)
    reasoning_tokens: int = field(default=0)

@dataclass
class LMConfig:
    """
    
    Args:
        provider: The provider of the language model
        
        cost_indicator: The cost indicator of the language model
            E.g. if you have model options: [llama-3b-fireworks, 4o-mini, 4o]
                you maye set the indocator for each option as [0.3, 1, 20]
            
        kwargs: The kwargs to initialize the language model
    """
    provider: Literal['openai', 'together', 'fireworks', 'local']
    model: str
    cost_indicator: float = field(default=1.0)
    kwargs: dict = field(default_factory=dict)
    price_table: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj
    
    def update(self, other: 'LMConfig'):
        """update the lm_config of the module
        
        passed in model_config should not be changed
        """
        self.provider = other.provider
        self.model = other.model
        self.cost_indicator = other.cost_indicator
        self.kwargs.update(other.kwargs)
        self.price_table.update(other.price_table)

    def get_price(self, usage: TokenUsage):
        if self.provider == 'local':
            return 0.0
        prompt, completion = usage.prompt_tokens, usage.completion_tokens
        model = self.model
        if self.provider == 'openai':
            prompt_cached_tokens = usage.prompt_cached_tokens
            prompt -= prompt_cached_tokens
            if 'gpt-4o-mini' in model:
                return (0.15 * prompt +  0.6 * completion + 0.075 * prompt_cached_tokens) / 1e6
            elif 'gpt-4o-2024-05-13' in model:
                return (5 * prompt + 15 * completion + 5 * prompt_cached_tokens) / 1e6
            elif 'gpt-4o-audio' in model:
                return (2.5 * prompt + 10 * completion + 2.5 * prompt_cached_tokens) / 1e6 
            elif 'gpt-4o' in model:
                return (2.5 * prompt + 10 * completion + 1.25 * prompt_cached_tokens) / 1e6
        elif self.provider == 'together':
            if 'meta-llama/Llama-3.2-3B-Instruct-Turbo' in model:
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
        elif self.provider == 'fireworks':
            if 'accounts/fireworks/models/llama-v3p2-3b-instruct' in model:
                return 0.1 * (prompt + completion) / 1e6
            elif 'llama-v3p1-8b-instruct' in model:
                return 0.2 * (prompt + completion) / 1e6
        
        raise ValueError(f"Model {model} from provider {self.provider} pricing is not supported")
            
        
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
    
    def __repr__(self):
        """Naive string representation of the demonstration
        
        NOTE: This is not used for adding demo to the actual user message. check `add_demos_to_prompt` instead.
        
        Semantic designner should have their own way to present the demonstration
        especially when the input contains other modalities
        """
        
        def truncate(text, max_length=200):
            """Truncate text if it exceeds max_length, appending '...' at the end."""
            return text if len(text) <= max_length else text[:max_length] + "..."
        
        inputs_trunced = {k: truncate(v) for k, v in self.inputs.items()}
        input_str = '**Input**\n' + json.dumps(inputs_trunced, indent=4)
        if self.reasoning:
            input_str += f"\n\n**Reasoning**\n{truncate(self.reasoning)}"
        demo_str = f"{input_str}\n\n**Response**\n{truncate(self.output)}"
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
 

_thread_local_chain = threading.local()

def _local_forward(_local_lm: 'LLMPredictor', **kwargs):
    _local_lm.input_cache = copy.deepcopy(kwargs)
    if _local_lm.lm is None:
        # if lm is reset or not set, initialize it and the kernel
        _local_lm.set_lm()
        _local_lm.initialize_kernel()
        
    if _local_lm.kernel is None:
        _local_lm.initialize_kernel()
    
    result = _local_lm.kernel(**kwargs)
    
    lm_hist = _local_lm.get_lm_history()
    _local_lm.step_info.append({
        'inputs': _local_lm.input_cache,
        'rationale': _local_lm.rationale,
        'output': lm_hist[-1]['response'],
    })
    _local_lm.rationale = None
    _local_lm.input_cache = {}
    _local_lm.lm_history.extend(lm_hist)
    
    return result
    
class LLMPredictor(Module):
    def __init__(self, name, semantic: LMSemantic, lm, lm_config, **kwargs) -> None:
        self.lm_history = []
        self.lm_config: LMConfig = lm_config
        self.lm = lm
        self.input_cache = {}
        self.step_info = []
        self.rationale: str = None
        self._lock = threading.Lock()
        
        self.semantic = semantic
        # NOTE: lm and kernel will be set at first execution
        # this is to allow deepcopy of the module
        super().__init__(name=name, kernel=None, **kwargs)
        self.input_fields = self.semantic.get_agent_inputs()
        setattr(_thread_local_chain, name, self)
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != '_lock':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, threading.Lock())
        return result
        
    @property
    def lm(self):
        return self._lm
    
    @lm.setter
    def lm(self, value):
        self._lm = value

    def get_thread_local_chain(self):
        try:
            if not hasattr(_thread_local_chain, self.name):
                # NOTE: no need to set to local storage bc that's mainly used to detect if current context is in a new thread
                _self = copy.deepcopy(self)
                _self.reset()
            else:
                _self = getattr(_thread_local_chain, self.name)
            return _self
        except Exception as e:
            logger.info(f'Error in get_thread_local_chain: {e}')
            raise
    
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
        Optional: {
            'prompt_cached_tokens': int,
            'reasoning_tokens': int,
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

    def get_token_usage(self) -> TokenUsage:
        """get current token usage of the LLM
        
        Please reset the usage cache at your will
        """
        #NOTE: a LLMPredictor might have multiple LLMs in its history
        # if the config is dynamically changing
        usage = TokenUsage()
        logger.debug(f"{self.name} meta len: {len(self.lm_history)}, {len(self.step_info)}")
        for meta in self.lm_history:
            # log tokens
            prompt_tokens = meta['prompt_tokens']
            completion_tokens = meta['completion_tokens']
            prompt_cached_tokens = meta.get('prompt_cached_tokens', 0)
            reasoning_tokens = meta.get('reasoning_tokens', 0)
            
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            usage.prompt_cached_tokens += prompt_cached_tokens
            usage.reasoning_tokens += reasoning_tokens
        return usage

    def get_total_cost(self) -> float:
        usage = self.get_token_usage()
        price = self.lm_config.get_price(usage)
        logger.debug(f"Token usage {self.name}: {usage}, price: {price}")
        return price
        
    def forward(self, **kwargs):
        _self = self.get_thread_local_chain()
        result = _local_forward(_self, **kwargs)
        self.aggregate_thread_local_meta(_self)
        return result

    def aggregate_thread_local_meta(self, _local_self):
        if self is _local_self:
            return
        with self._lock:
            self.step_info.extend(_local_self.step_info)
            self.lm_history.extend(_local_self.lm_history)