import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
import copy
import logging
import optunahub
import optuna
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
import math
import threading
import uuid
from dataclasses import dataclass, field
import multiprocessing as mp

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase, DynamicParamBase, EvolveType, AddNewModuleImportInterface
from compiler.optimizer.params.utils import dump_params, load_params
from compiler.optimizer.params.model_selection import LMSelection
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask, GeneralEvaluatorInterface
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from compiler.optimizer.plugin import OptimizerSchema
from optuna.samplers import TPESampler
from compiler.optimizer.core.flow import TrialLog, ModuleTransformTrace, TopDownInformation, OptConfig
from compiler.optimizer.core.unified_layer_opt import OptimizationLayer, BottomLevelOptimization, UpperLevelOptimization, LayerEvaluator

class layerConfig:
    def __init__(
        self,
        layer_name: str,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 0,
        opt_config: OptConfig = None,
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
        

class MultiLayerOptimizationDriver:
    def __init__(
        self,
        layer_configs: Sequence[layerConfig],
        evaluator: EvaluatorPlugin,
        quality_constraint: float = None,
    ):
        """Driver for multi-layer optimization
        
        Args:
            layer_configs (Sequence[layerConfig]): configs for each optimization layer
            
        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        self.layer_configs = layer_configs
        self.evaluator = evaluator
        self.quality_constraint = quality_constraint
        
        # initialize optimization layers
        self.opt_layers: list[OptimizationLayer] = [None] * len(layer_configs)
        self.build_tiered_optimization()
    
    def build_tiered_optimization(
        self,
    ):
        """Build tiered optimization from bottom to top
        """
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - 1 - ri
            if ri == 0:
                opt_layer = BottomLevelOptimization(
                    name=layer_config.layer_name,
                    evaluator=self.evaluator,
                    dedicate_params=layer_config.dedicate_params,
                    universal_params=layer_config.universal_params,
                    target_modules=layer_config.target_modules,
                    save_ckpt_interval=layer_config.save_ckpt_interval,
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
                )
            self.opt_layers[idx] = opt_layer
            
    def run(
        self, 
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[TrialLog], dict[int, TrialLog]]:
        first_layer_opt_config = self.layer_configs[0].opt_config
        return self.opt_layers[0].easy_optimize(
            opt_config=first_layer_opt_config,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
    
    def evaluate(
        self,
        bot_trial_log_id: str,
        opt_log_path: str,
    ):
        bot_layer: BottomLevelOptimization = self.opt_layers[-1]
        return bot_layer.easy_eval(
            trial_log_id=bot_trial_log_id,
            opt_log_path=opt_log_path,
        )
        