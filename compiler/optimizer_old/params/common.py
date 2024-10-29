from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, auto
from collections import defaultdict
from typing import Type, Union
import logging
import json

from compiler.IR.base import Module
from compiler.patterns.blocker import Protection

logger = logging.getLogger(__name__)

class OptionBase(ABC):
    def __init__(self, name: str):
        self.name = name
    
    def _get_cost_indicator(self):
        return 1.0
    
    @property
    def cost_indicator(self):
        return self._get_cost_indicator()
    
    @abstractmethod
    def apply(self, module: Module) -> Module:
        """Apply the option to the module
        
        Please do not seek information from the enclosing module
        as the provided module might just be a dangling module and will be integrated into the workflow later
        """
        ...
    
    def exposed_apply(self, module: Module) -> Module:
        try:
            with Protection(module, ['enclosing_module', 'get_immediate_enclosing_module']):
                return self.apply(module)
        except Exception as e:
            logger.error(f'Error in applying {self.name} to {module.name}')
            logger.error(e)
            raise
    
    def to_dict(self):
        return {
            'name': self.name,
            'type': self.__class__.__name__,
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        data.pop('type', None)
        return cls(**data)

    def obtain_cost_indicator(self) -> float:
        return self.cost_indicator

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
    level_to_params = defaultdict(list)
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
            cls.level_to_params[level].append(new_cls)
        return new_cls

T_ModuleMapping = dict[str, str] 
"""mapping from old module name to new module name

NOTE: this can be recursive if we support evolutioanry optimization
Example:
    orignal workflow: [a, b, c]
    mapping: 
        a -> [a1]
        a1 -> [a2]
        c -> [c1]
"""

# TODO: inplace merge if needed
def mmerge(mapping1: T_ModuleMapping, mapping2: T_ModuleMapping) -> T_ModuleMapping:
    """create a new mapping by merging two mappings
    """
    result = defaultdict(list, mapping1)  
    for key, value in mapping2.items():
        result[key].extend(value)
    return result

# TODO: support cycles
def mflatten(mapping: T_ModuleMapping) -> T_ModuleMapping:
    """flatten the mapping
    
    Example:
        original: 
            a -> [a1]
            a1 -> [a2]
        after:
            a -> [a2]
    """
    def is_cyclic_util(graph, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True
        
        if v in graph:
            neighbor = graph[v]
            if not visited[neighbor]:
                if is_cyclic_util(graph, neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True
            
        rec_stack[v] = False
        return False

    visited = defaultdict(False.__bool__)
    rec_stack = defaultdict(False.__bool__)
    for key in mapping:
        if not visited[key]:
            if is_cyclic_util(mapping, key, visited, rec_stack):
                raise ValueError('Cyclic mapping')
    
    result: T_ModuleMapping = {}
    for key, value in mapping.items():
        while value in mapping:
            value = mapping[value]
        result[key] = value
    return result

class AddNewModuleImportInterface:
    @abstractmethod
    def get_python_paths(self) -> list[str]:
        """return a list of python paths to be added to the PYTHON_PATH
        
        This is required when new modules can be generated during the optimization process
        """
        ...
    
class ParamBase(metaclass=ParamMeta):
    ParamMeta.required_fields = ['level']
    
    def __init__(
        self, 
        name: str, 
        options: list[OptionBase],
        default_option: Union[int, str] = 0,
        module_name: str = None,
        inherit: bool = False,
    ):
        """Define a parameter with a list of options
        
        Args:
            name: the name of the parameter
            
            options: a list of options
            
            default_option: the default option index
            
            module_name: the name of the module to apply the option
                set to None if this param is universal
                
            inherit: whether the param can be inherited by new modules that are created from the current module
            
                If module_name is None, the param is universally applied so this flag will be ignored
                
                By default all options are inherited, if you are using DynamicParamBase, you can set inherit_options to control this
        """
        self.name = name
        self.module_name = module_name
        self.options: dict[str, OptionBase] = {option.name: option for option in options}
        if isinstance(default_option, int):
            self.default_option: str = options[default_option].name
        else:
            self.default_option = default_option
        self.inherit = inherit
    
    def get_default_option(self):
        return self.options[self.default_option]
    
    @staticmethod
    def chash(module_name, param_name):
        return f"{module_name}_{param_name}"
    
    @property
    def hash(self):
        return ParamBase.chash(self.module_name, self.name)
       
    def apply_option(self, option: str, module: Module) -> tuple[Module, T_ModuleMapping]:
        """Apply the idx-th option to the module
        
        Will change the module in-place or return a new one and will replace it in the enclosing module
        
        Beside the new module, return a mapping from old module name to new module name
        """
        assert module is not None, f'Param {self.name} has no module to apply'
        assert option in self.options, f'Option {option} not found in {self.options.keys()}'
        
        old_name = module.name
        new_module = self.options[option].exposed_apply(module)
        
        # populate mapping
        mapping: T_ModuleMapping = {}
        if new_module.name != old_name:
            mapping[old_name] = new_module.name
            
        module.chameleon(new_module)
        return new_module, mapping

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
    
    def __init__(
        self, 
        name: str, 
        options: list[OptionBase], 
        default_option: int | str = 0, 
        module_name: str = None, 
        inherit: bool = False,
        inherit_options: bool = False,
    ):
        """
        
        Args:
            inherit_options: whether the options of this param can be inherited by new modules
            
        """
        super().__init__(name, options, default_option, module_name, inherit)
        self.inherit_options = inherit_options
    
    def add_option(self, option: OptionBase):
        if option.name in self.options:
            Warning(f'Rewriting Option {option.name} in param {self.hash}')
        self.options[option.name] = option
    
    def clean_state(self):
        """Routine to clean up the state of the param for non-inherited options
        """
        self.custom_clean()
        self.options = {
            name: option for name, option in self.options.items() 
            if option.name == 'Identity'
        }
    
    @abstractmethod
    def custom_clean(self):
        ...
    
    @abstractmethod
    def evolve(self, eval_result) -> EvolveType:
        ...
        