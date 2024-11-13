import dataclasses
import os
import json
import importlib

from cognify.optimizer.core.driver import LayerConfig
from cognify.optimizer.plugin import capture_module_from_fs

@dataclasses.dataclass
class ControlParameter:
    opt_layer_configs: list[LayerConfig]
    opt_history_log_dir: str = 'opt_results'
    quality_constraint: float = 1.0
    train_down_sample: int = 0
    val_down_sample: int = 0
    evaluator_batch_size: int = 20
    
    @classmethod
    def build_control_param(cls, param_path=None, loaded_module=None):
        assert param_path or loaded_module, "Either param_path or loaded_module should be provided."
        
        if param_path:
            if not os.path.isfile(param_path):
                raise FileNotFoundError(f"The control param file {param_path} does not exist.")
            module_name = os.path.basename(param_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, param_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = loaded_module
        
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, cls):
                return obj
        raise ValueError(f"No ControlParameter instance found in {param_path}")
    
    def __post_init__(self):
        # create directory for logging
        if not os.path.exists(self.opt_history_log_dir):
            os.makedirs(self.opt_history_log_dir, exist_ok=True)
        