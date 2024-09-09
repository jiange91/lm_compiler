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

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase
from compiler.langchain_bridge.interface import LangChainLM


logger = logging.getLogger(__name__)

# class BaseLOPT(ABC):
#     """One layer of optimization
    
#     Building block of hierarchical optimization
#     """
#     def __init__(
#         self,
#         modules: list[Module],
#         params: dict[str, list[ParamBase]],
#         final_output_metric: Callable[[Any, StatePool], Any],
#         opt_direction: Literal['maximize', 'minimize'],
#     ):
#         """Create a LOPT instance with configs
        
#         Args:
#             params: a dict mapping param name to a list of 
#         """
#         self.modules = modules
#         self.params = params
#         self.final_output_metric = final_output_metric
#         self.opt_direction = opt_direction

    # @abstractmethod
    # def transform():
    #     pass
    
    # @abstractmethod
    # def step():
    #     pass

def param_hash(param: ParamBase):
    return f"{param.module_name}_{param.name}"

class InnerLoopBayesianOptimization:
    def __init__(
        self,
        params: Union[dict[str, list[ParamBase]], list[ParamBase]],
        opt_direction: Literal['maximize', 'minimize'],
        target_modules: Iterable[str] = None,
        fields_in_interest: list[str] = None
    ):
        """
        
        Args:
            params: if a list is provided, it will be broadcasted to all modules
                    for each module, params are applied in order
            
            target_modules: if provided, only the modules in this list will be optimized
        """
        self.raw_params = params
        self.opt_direction = opt_direction
        self.fields_in_interest = fields_in_interest
        self.target_modules = set(target_modules) if target_modules is not None else None
    
    def prepare_params(self, lm_modules: list[LangChainLM]):
        self.params: dict[str, list[ParamBase]] = {}
        if isinstance(self.raw_params, list):
            for lm in lm_modules:
                lm_name = lm.name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                params_cpy = copy.deepcopy(self.raw_params)
                for param in params_cpy:
                    param.module_name = lm_name
                self.params[lm_name] = params_cpy
        else:
            for lm_name, params in self.raw_params.items():
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                for param in params:
                    assert param.module_name == lm_name, f"param {param.name} has module_name {param.module_name} not matching {lm_name}"
                self.params[lm_name] = self.raw_params[lm_name]
        self.param_categorical_dist = {
            param_hash(param): list(range(len(param.options)))
            for _, params in self.params.items() for param in params
        }
        self.tpe_logs = {}
    
    # TODO: support more than langchain
    def get_objective_function(self, evaluator, workflow: Workflow):
        def objective_function(trial: optuna.Trial):
            candidate = copy.deepcopy(workflow)
            lm_modules = candidate.get_all_modules(lambda x: isinstance(x, LangChainLM))
            module_dict: dict[str, LangChainLM] = {lm.name: lm for lm in lm_modules}
            
            logger.info(f"innerloop - start Trial: {trial.number}")
            self.tpe_logs[trial.number] = {}
            self.tpe_logs[trial.number]['params'] = {}
            
            for lm_name, params in self.params.items():
                for param in params:
                    id = param_hash(param)
                    selected = trial.suggest_categorical(id, self.param_categorical_dist[id])
                    new_lm = param.apply_option(selected, module_dict[lm_name])
                    module_dict[lm_name] = new_lm
                    self.tpe_logs[trial.number]['params'][id] = param.options[selected].name
            logger.info(f"innerloop - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}") 
            
            states, score, price = evaluator(candidate)
            self.tpe_logs[trial.number]['score'] = score
            self.tpe_logs[trial.number]['price'] = price
            logger.info(f"innerloop - Trial {trial.number} result: score: {score}, price: {price}")
            
            if self.fields_in_interest is not None:
                self.tpe_logs[trial.number]['fields'] = [state.all_news(self.fields_in_interest) for state in states]
            return score, price 
        return objective_function
        
    def optimize(
        self,
        workflow: Workflow,
        evaluator: Callable[[Workflow], Tuple[Iterable[StatePool], Union[int, float], float]],
        n_trials: int,
        log_dir: str = 'holm_log',
    ):
        """Find optimal params for the given workflow
        
        Will not modify the original workflow
        """
        self.inner_loop_log_dir = os.path.join(log_dir, 'inner_loop')
        if not os.path.exists(self.inner_loop_log_dir):
            os.makedirs(self.inner_loop_log_dir, exist_ok=True)
        
        workflow.reset()
        self.prepare_params(workflow.get_all_modules(lambda x: isinstance(x, LangChainLM)))
        obj_func = self.get_objective_function(evaluator=evaluator, workflow=workflow)
        study = optuna.create_study(directions=[self.opt_direction, 'minimize']) # minimize for price
        
        study.optimize(obj_func, n_trials=n_trials)
            
        json.dump(self.tpe_logs, open(os.path.join(self.inner_loop_log_dir, 'tpe_logs.json'), 'w+'), indent=4)
        
        for i, trial in enumerate(study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            mapped_params = {name, }
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: f1={}, f2={}".format(f1, f2))
        return study.best_trials