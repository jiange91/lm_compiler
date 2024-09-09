import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal
import copy
import logging
import optunahub
import optuna
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase
from compiler.optimizer.layered_optimizer import InnerLoopBayesianOptimization
from compiler.optimizer.evaluation.evaluator import Evaluator

class LayerDriver:
    def __init__(
        self,
        workflow: Workflow,
        layer: InnerLoopBayesianOptimization,
        evaluator: Evaluator,
    ) -> None:
        self.workflow = workflow
        self.layer = layer
        self.evalutor = evaluator
        
    def fire(
        self,
        log_dir: str = 'layer_driver_logs',
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        self.layer.optimize(
            self.workflow, self.evalutor, 10, log_dir
        )