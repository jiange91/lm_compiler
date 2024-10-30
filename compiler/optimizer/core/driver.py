import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
from dataclasses import dataclass, field, asdict
import logging

from compiler.optimizer.params.common import ParamBase
from compiler.optimizer.params.utils import build_param
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask, GeneralEvaluatorInterface
from compiler.optimizer.core.flow import TrialLog, OptConfig
from compiler.optimizer.core.unified_layer_opt import OptimizationLayer, BottomLevelOptimization, BottomLevelTrialLog
from compiler.optimizer.core.upper_layer import UpperLevelOptimization, LayerEvaluator

class LayerConfig:
    def __init__(
        self,
        layer_name: str,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 1,
        opt_config: OptConfig = None,
        use_SH_allocation: bool = True,
    ):
        """Config for each optimization layer
        
        Args:
            layer_name (str): name of the layer
            
            dedicate_params (list[ParamBase], optional): dedicated params for this layer. Defaults to [].
            
            universal_params (list[ParamBase], optional): universal params for this layer. Defaults to [].
            
            target_modules (Iterable[str], optional): target modules for this layer. Defaults to None.
            
            save_ckpt_interval (int, optional): save checkpoint interval. Defaults to 0.
            
            opt_config (OptConfig, optional): optimization config. Defaults to None.
                all fields not set here will be decided by the upper layer
            
            use_SH_allocation (bool, optional): whether to use SH allocation. Defaults to False.
        """
        self.layer_name = layer_name
        self.dedicate_params = dedicate_params
        self.universal_params = universal_params
        self.target_modules = target_modules
        self.save_ckpt_interval = save_ckpt_interval
        self.opt_config = opt_config
        self.use_SH_allocation = use_SH_allocation
        
        if len(self.dedicate_params) + len(self.universal_params) == 0:
            raise ValueError(f'No params provided for optimization layer {layer_name}')
    
    def to_dict(self):
        return {
            'layer_name': self.layer_name,
            'dedicate_params': [p.to_dict() for p in self.dedicate_params],
            'universal_params': [p.to_dict() for p in self.universal_params],
            'target_modules': self.target_modules,
            'save_ckpt_interval': self.save_ckpt_interval,
            'opt_config': asdict(self.opt_config),
            'use_SH_allocation': self.use_SH_allocation,
        }
    
    @classmethod
    def from_dict(cls, d):
        return cls(
            layer_name=d['layer_name'],
            dedicate_params=[build_param(p) for p in d['dedicate_params']],
            universal_params=[build_param(p) for p in d['universal_params']],
            target_modules=d['target_modules'],
            save_ckpt_interval=d['save_ckpt_interval'],
            opt_config=OptConfig(**d['opt_config']),
            use_SH_allocation=d['use_SH_allocation'],
        )
        

class MultiLayerOptimizationDriver:
    def __init__(
        self,
        layer_configs: Sequence[LayerConfig],
        opt_log_dir: str,
        quality_constraint: float = None,
        save_config_to_file: bool = True,
    ):
        """Driver for multi-layer optimization
        
        Args:
            layer_configs (Sequence[layerConfig]): configs for each optimization layer
            
        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        self.layer_configs = layer_configs
        self.quality_constraint = quality_constraint
        
        # initialize optimization layers
        self.opt_layers: list[OptimizationLayer] = [None] * len(layer_configs)
        
        # dump control params
        if not os.path.exists(opt_log_dir):
            os.makedirs(opt_log_dir, exist_ok=True)
        param_log_path = os.path.join(opt_log_dir, 'opt_control_params.json')
        layer_configs_dict = [lc.to_dict() for lc in layer_configs]
        if save_config_to_file:
            with open(param_log_path, 'w') as f:
                json.dump(layer_configs_dict, f, indent=4)
        
        # config log dir for layer opts
        # NOTE: only the top layer will be set, others are decided at runtime
        self.layer_configs[0].opt_config.log_dir = os.path.join(opt_log_dir, self.layer_configs[0].layer_name)
    
    def build_tiered_optimization(
        self, evaluator: EvaluatorPlugin
    ):
        """Build tiered optimization from bottom to top
        """
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - 1 - ri
            if ri == 0:
                opt_layer = BottomLevelOptimization(
                    name=layer_config.layer_name,
                    evaluator=evaluator,
                    dedicate_params=layer_config.dedicate_params,
                    universal_params=layer_config.universal_params,
                    target_modules=layer_config.target_modules,
                    save_ckpt_interval=layer_config.save_ckpt_interval,
                    quality_constraint=self.quality_constraint,
                )
            else:
                layer_evaluator = LayerEvaluator(
                    target_layer=self.opt_layers[idx + 1],
                    quality_constraint=self.quality_constraint,
                )
                opt_layer = UpperLevelOptimization(
                    name=layer_config.layer_name,
                    evaluator=layer_evaluator,
                    dedicate_params=layer_config.dedicate_params,
                    universal_params=layer_config.universal_params,
                    target_modules=layer_config.target_modules,
                    save_ckpt_interval=layer_config.save_ckpt_interval,
                    next_level_opt_config=self.layer_configs[idx + 1].opt_config,
                    use_SH_allocation=layer_config.use_SH_allocation,
                    quality_constraint=self.quality_constraint,
                )
            self.opt_layers[idx] = opt_layer
            
    def run(
        self,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[TrialLog], dict[int, TrialLog]]:
        self.build_tiered_optimization(evaluator)
        first_layer_opt_config = self.layer_configs[0].opt_config
        return self.opt_layers[0].easy_optimize(
            opt_config=first_layer_opt_config,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
    
    def inspect(self) -> tuple[float, list[TrialLog], dict[int, TrialLog]]:
        self.build_tiered_optimization(None)
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        result = self.opt_layers[0].inspect(opt_config.opt_log_path)
        return result
        