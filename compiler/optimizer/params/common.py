from abc import ABC, abstractmethod
from enum import Enum, auto

class ParamLevel(Enum):
    GRAPH = auto()
    NODE = auto()
    
class ParamDimension(Enum):
    REASONING = auto()
    EXAMPLES = auto()
    INSTS = auto()
    
param_pool = {}

    
class ParamSpec:
    def __init__(
        self, 
        name: str,
        level: ParamLevel, 
        dimension: ParamDimension, 
    ):
        self.name = name
        self.level = level
        self.dimension = dimension