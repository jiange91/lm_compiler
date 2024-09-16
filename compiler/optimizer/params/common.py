from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, auto
from collections import defaultdict
from typing import Type, Union
import logging
import json

from compiler.IR.base import Module
from compiler.optimizer.evaluation.evaluator import EvaluationResult

class OptionBase(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, module: Module) -> Module:
        ...
    
    def to_dict(self):
        return {
            'name': self.name,
            'type': self.__class__.__name__,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        data.pop('type', None)
        return cls(**data)

class IdentityOption(OptionBase):
    def __init__(self):
        super().__init__('Identity')
    
    def apply(self, module: Module) -> Module:
        return module

    @classmethod
    def from_dict(cls, data: dict):
        return cls()
    
class ParamLevel(Enum):
    """Please do not change this 
    the optimizer is not generic to any number of levels
    """
    GRAPH = auto()
    NODE = auto()
    
class ParamMeta(ABCMeta):
    required_fields = []
    registry = {}
    level_2_params = defaultdict(list)
    base_class = {'ParamBase', 'DynamicParamBase'}
    
    def __new__(cls, name, bases, attrs):
        if name not in cls.base_class:
            # TODO: add more checkings
            for field in cls.required_fields:
                if field not in attrs:
                    raise ValueError(f'{name} must have {field}')
            level = attrs['level'] 
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        if name not in cls.base_class:
            cls.level_2_params[level].append(new_cls)
        return new_cls

class ParamBase(metaclass=ParamMeta):
    ParamMeta.required_fields = ['level']
    
    def __init__(
        self, 
        name: str, 
        options: list[OptionBase],
        default_option: Union[int, str] = 0,
        module_name: str = None,
    ):
        self.name = name
        self.module_name = module_name
        self.options: dict[str, OptionBase] = {option.name: option for option in options}
        if isinstance(default_option, int):
            self.default_option: str = options[default_option].name
        else:
            self.default_option = default_option
    
    @property
    def hash(self):
        return f"{self.module_name}_{self.name}"
       
    def apply_option(self, option: str, module: Module) -> Module:
        """Apply the idx-th option to the module
        
        Change the module in-place or replace it with a new module
        """
        assert module is not None, f'Param {self.name} has no module to apply'
        assert option in self.options, f'Option {option} not found in {self.options.keys()}'
        
        new_module = self.options[option].apply(module)
        if new_module is not module:
            if module.enclosing_module is not None:
                # NOTE: in-place replacement
                module.enclosing_module.replace_node(module, new_module, new_module)
        return new_module

    def to_dict(self):
        return {
            'name': self.name,
            'module_name': self.module_name,
            'options': {name: option.to_dict() for name, option in self.options.items()},
            'default_option': self.default_option,
            'type': self.__class__.__name__,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class EvolveType(Enum):
    ID = auto() # no change
    RANGE = auto() # range of values
    ENTITY = auto() # add or remove entities

class DynamicParamBase(ParamBase, ABC):
    
    def add_option(self, option: OptionBase):
        if option.name in self.options:
            Warning(f'Rewriting Option {option.name} in param {self.hash}')
        self.options[option.name] = option
    
    @abstractmethod
    def evole(self, eval_result: EvaluationResult) -> EvolveType:
        ...

class LMDecompose(ParamBase):
    level = ParamLevel.GRAPH

