from cognify.optimizer.core import driver
from cognify.llm import LMConfig
from abc import ABC

class BaseOptimSetup(ABC):
    def __init__(self, layer_configs: list[driver.LayerConfig], throughput: int = 2):
        self.layer_configs = layer_configs
        self.throughput = throughput

class OptimSetup(BaseOptimSetup):
    def __init__(self, layer_configs: list[driver.LayerConfig], throughput: int = 2):
        super().__init__(layer_configs, throughput)

class OptimSetupWithModelSelection(BaseOptimSetup):
    def __init__(self, layer_configs: list[driver.LayerConfig], model_configs: list[LMConfig], throughput: int = 1):
        super().__init__(layer_configs, throughput)
        self.model_configs = model_configs