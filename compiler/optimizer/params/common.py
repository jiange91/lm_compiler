from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, auto
from collections import defaultdict
from typing import Type

from compiler.IR.base import Module

class OptionBase(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, module: Module) -> Module:
        ...

class IdentityOption(OptionBase):
    def __init__(self):
        super().__init__('Identity')
    
    def apply(self, module: Module) -> Module:
        return module
    
class ParamLevel(Enum):
    """Please do not change this 
    the optimizer is not generic to any number of levels
    """
    GRAPH = auto()
    NODE = auto()
    
class ParamMeta(ABCMeta):
    required_fields = []
    param_pool = defaultdict(list)
    
    def __new__(cls, name, bases, attrs):
        if name != 'ParamBase':
            # TODO: add more checkings
            for field in cls.required_fields:
                if field not in attrs:
                    raise ValueError(f'{name} must have {field}')
            level = attrs['level'] 
        new_cls = super().__new__(cls, name, bases, attrs)
        if name != 'ParamBase':
            cls.param_pool[level].append(new_cls)
        return new_cls

class ParamBase(metaclass=ParamMeta):
    ParamMeta.required_fields = ['level']
    
    def __init__(self, name: str, module_name: str, options: list[OptionBase]):
        self.name = name
        self.module_name = module_name
        self.options = options
       
    def apply_option(self, idx, module: Module) -> Module:
        """Apply the idx-th option to the module
        
        Change the module in-place or replace it with a new module
        """
        assert module is not None, f'Param {self.name} has no module to apply'
        new_module = self.options[idx].apply(module)
        if new_module is not module:
            if module.enclosing_module is not None:
                # NOTE: in-place replacement
                module.enclosing_module.replace_node(module, new_module, new_module)
        return new_module


class LMDecompose(ParamBase):
    level = ParamLevel.GRAPH


class LMFewShot(ParamBase):
    level = ParamLevel.NODE

