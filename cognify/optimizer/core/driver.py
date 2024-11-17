import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
from dataclasses import dataclass, field, asdict
import logging
import re

from cognify.cog_hub.common import CogBase
from cognify.cog_hub.utils import build_param
from cognify.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from cognify.optimizer.core.flow import TrialLog, OptConfig
from cognify.optimizer.core.unified_layer_opt import OptimizationLayer, BottomLevelOptimization, BottomLevelTrialLog
from cognify.optimizer.core.upper_layer import UpperLevelOptimization, LayerEvaluator
import logging

logger = logging.getLogger(__name__)

class LayerConfig:
    def __init__(
        self,
        layer_name: str,
        dedicate_params: list[CogBase] = [],
        universal_params: list[CogBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 1,
        opt_config: Optional[OptConfig] = None,
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
            
        """
        self.layer_name = layer_name
        self.dedicate_params = dedicate_params
        self.universal_params = universal_params
        self.target_modules = target_modules
        self.save_ckpt_interval = save_ckpt_interval
        self.opt_config = opt_config
        
        if len(self.dedicate_params) + len(self.universal_params) == 0:
            raise ValueError(f'No params provided for optimization layer {layer_name}')
        
        if self.opt_config is None:
            self.opt_config = OptConfig(n_trials=5)
    
    def to_dict(self):
        return {
            'layer_name': self.layer_name,
            'dedicate_params': [p.to_dict() for p in self.dedicate_params],
            'universal_params': [p.to_dict() for p in self.universal_params],
            'target_modules': self.target_modules,
            'save_ckpt_interval': self.save_ckpt_interval,
            'opt_config': asdict(self.opt_config),
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
            layer_configs (Sequence[LayerConfig]): configs for each optimization layer
            
        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        self.layer_configs = layer_configs
        self.quality_constraint = quality_constraint
        
        # initialize optimization layers
        self.opt_layers: list[OptimizationLayer] = [None] * len(layer_configs)
        
        # dump control params
        self.opt_log_dir = opt_log_dir
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
                    hierarchy_level=idx,
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
                    use_SH_allocation=layer_config.opt_config.use_SH_allocation,
                    quality_constraint=self.quality_constraint,
                    hierarchy_level=idx,
                )
            self.opt_layers[idx] = opt_layer
            
    def run(
        self,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[tuple[TrialLog, str]], dict[str, TrialLog]]:
        self.build_tiered_optimization(evaluator)
        first_layer_opt_config = self.layer_configs[0].opt_config
        logger.info("----------------- Start Optimization -----------------")
        opt_cost, frontier, all_opt_logs = self.opt_layers[0].easy_optimize(
            opt_config=first_layer_opt_config,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        logger.info("----------------- Optimization Finished -----------------")
        self.dump_frontier_details(frontier)
        return opt_cost, frontier, all_opt_logs
    
    def _extract_trial_id(self, config_id: str) -> str:
        param_log_dir = os.path.join(self.opt_log_dir, 'pareto_frontier_details')
        if not os.path.exists(param_log_dir):
            raise ValueError(f"Cannot find the optimization log directory at {param_log_dir}")
        
        with open(os.path.join(param_log_dir, f"{config_id}.cog"), 'r') as f:
            first_line = f.readline().strip()
        match = re.search(r"Trial - (.+)", first_line)
        if match:
            trial_id = match.group(1)
            return trial_id
        else:
            raise ValueError(f"Cannot extract trial id from the log file {config_id}.cog")
    
    def _find_config_log_path(self, trial_id: str) -> str:
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        
        top_layer = self.opt_layers[0]
        top_layer.load_opt_log(opt_config.opt_log_path)
        all_configs = top_layer.get_all_candidates(opt_config.opt_log_path)
        config_path = None
            
        for opt_log, path in all_configs:
            if opt_log.id == trial_id:
                config_path = path
                break
        else:
            raise ValueError(f"Config {trial_id} not found in the optimization log.")
        return config_path

    def evaluate(
        self,
        evaluator: EvaluatorPlugin,
        config_id: str,
    ) -> EvaluationResult:
        self.build_tiered_optimization(evaluator)
        trial_id = self._extract_trial_id(config_id)
        config_path = self._find_config_log_path(trial_id)
        
        result = BottomLevelOptimization.easy_eval(
            evaluator=evaluator,
            trial_id=trial_id,
            opt_log_path=config_path,
        )
        return result
    
    def load(
        self,
        config_id: str,
    ):
        self.build_tiered_optimization(None)
        trial_id = self._extract_trial_id(config_id)
        config_path = self._find_config_log_path(trial_id)
        
        with open(config_path, 'r') as f:
            opt_trace = json.load(f)
        trial_log = BottomLevelTrialLog.from_dict(opt_trace[trial_id])
        eval_task = EvalTask.from_dict(trial_log.eval_task)
        schema, old_name_2_new_module = eval_task.load_and_transform()
        return schema, old_name_2_new_module
        
    
    def inspect(self, dump_details: bool = False):
        self.build_tiered_optimization(None)
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        
        self.opt_layers[0].load_opt_log(opt_config.opt_log_path)
        frontier = self.opt_layers[0].post_optimize()
        
        # dump frontier details to file
        if dump_details:
            self.dump_frontier_details(frontier)
        return 
    
    def dump_frontier_details(self, frontier):
        param_log_dir = os.path.join(self.opt_log_dir, 'pareto_frontier_details')
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir, exist_ok=True)
        for i, (trial_log, opt_path) in enumerate(frontier):
            trial_log: BottomLevelTrialLog
            dump_path = os.path.join(param_log_dir, f'Pareto_{i+1}.cog')
            trans = trial_log.show_transformation()
            details = f"Trial - {trial_log.id}\n"
            details += f"Log at: {opt_path}\n"
            details += f"Quality: {trial_log.score:.3f}, Cost per 1K invocation ($): {trial_log.price * 1000:.2f} $\n"
            details += trans
            with open(dump_path, 'w') as f:
                f.write(details)
