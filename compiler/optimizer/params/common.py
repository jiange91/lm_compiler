from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, auto
from collections import defaultdict

from compiler.IR.base import Module

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
    
    def __init__(self, options: list['OptionBase']):
        self.options = options
        
    @abstractmethod
    def apply_option(self, idx, module: Module) -> Module:
        new_module = self.options[idx].apply(module)
        if new_module is not module:
            if module.enclosing_module is not None:
                # NOTE: in-place replacement
                module.enclosing_module.replace_node(module, new_module, new_module)
        return new_module

class OptionBase(ABC):
    @abstractmethod
    def apply(self, module: Module) -> Module:
        ...

class IdentityOption(OptionBase):
    def apply(self, module: Module) -> Module:
        return module

class LMDecompose(ParamBase):
    level = ParamLevel.GRAPH


class LMFewShot(ParamBase):
    level = ParamLevel.NODE

