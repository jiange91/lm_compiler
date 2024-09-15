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
import uuid
import operator
from collections import deque
from ConfigSpace.conditions import InCondition


logger = logging.getLogger(__name__)

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase
from compiler.optimizer.layered_optimizer import LayerBase, OptRoutineBase
from compiler.optimizer.evaluation.evaluator import Evaluator


class LayerDriver:
    def __init__(
        self,
        layers: list[LayerBase],
        budgets: list[int],
        evaluator: Evaluator,
        opt_direction: Literal['maximize', 'minimize'],
    ) -> None:
        # TODO: automate the budget selection
        self.layers = layers
        self.budgets = budgets
        self.evalutor = evaluator
        self.opt_direction = opt_direction
        self.opt_logs = {}
        self.opt_cost = 0
    
    def orchestrate_stack_friendly(
        self, workflow: Workflow
    ):
        pass
    
    def orchestrate(
        self, workflow: Workflow, layer_idx: int, params: dict,
    ):
        if layer_idx == len(self.layers):
            # bottom layer, run the evaluator
            _, score, price = self.evalutor(workflow)
            record_id = str(uuid.uuid4())
            self.opt_logs[record_id] = {}
            self.opt_logs[record_id]['score'] = score
            self.opt_logs[record_id]['price'] = price
            self.opt_logs[record_id]['params'] = params
            self.opt_cost += price
            return [(score, price)]
        
        # prepare the layer
        layer = self.layers[layer_idx]
        opt_routine = layer.prepare_params(workflow)
        results = []
        for i in range(self.budgets[layer_idx]):
            trials, candidates = opt_routine.propose(n_sample=1)
            new_params = params.copy()
            new_params.update(trials[0].params)
            score_price_list = self.orchestrate(
                candidates[0], 
                layer_idx+1,
                new_params,
            )
            opt_routine.update({trials[0].number: score_price_list})
            results.extend(score_price_list)
        return results

    def get_pareto_front(self):
        """
        Find the pareto-efficient points
        """
        record_ids = np.array(list(self.opt_logs.keys()))
        scale = -1 if self.opt_direction == 'maximize' else 1
        vectors = np.array([
            [scale*self.opt_logs[rid]['score'], self.opt_logs[rid]['price']] 
            for rid in record_ids]
        )
        is_efficient = np.ones(vectors.shape[0], dtype = bool)
        for i, v in enumerate(vectors):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(vectors[is_efficient]<v, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        masked = [
            self.opt_logs[rid] for rid in record_ids[is_efficient]
        ]
        return masked
                
    def fire(
        self,
        workflow: Workflow,
        log_dir: str = 'layer_driver_logs',
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        workflow.reset()
        opt_log_path = os.path.join(log_dir, 'opt_logs.json')
        if os.path.exists(opt_log_path):
            with open(opt_log_path, 'r') as f:
                self.opt_logs = json.load(f)
            for record_id, record in self.opt_logs.items():
                self.opt_cost += record['price']
        else:
            self.orchestrate(workflow, 0, {})
            with open(opt_log_path, 'w') as f:
                json.dump(self.opt_logs, f, indent=4)
        
        pareto_front = self.get_pareto_front()
        for record in pareto_front:
            print(record)
        return pareto_front