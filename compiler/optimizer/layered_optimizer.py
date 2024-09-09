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
from compiler.optimizer.params.model_selection import LMSelection
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.evaluator import Evaluator


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
        fields_in_interest: list[str] = None,
        important_lms: list[str] = None,
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
        self.important_lms = important_lms
        self.opt_cost = 0
    
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
        self.tpe_distributions = {
            id: optuna.distributions.CategoricalDistribution(options)
            for id, options in self.param_categorical_dist.items()
        }
        self.param_hash_2_options = {
            param_hash(param): [option.name for option in param.options]
            for _, params in self.params.items() for param in params
        }
        self.tpe_logs = {}
    
    def reduce_cold_start(self, study: optuna.Study):
        # set base config
        base_config = {key: 0 for key in self.param_categorical_dist}
        study.enqueue_trial(base_config)
        # create trials for important lms
        for lm_name, params in self.params.items():
            if lm_name in self.important_lms:
                for param in params:
                    if isinstance(param, LMSelection):
                        for i in range(1, len(param.options)):
                            config = copy.deepcopy(base_config)
                            config[param_hash(param)] = i
                            study.enqueue_trial(config)
        warm_start = len(study.get_trials(False))
        logger.info(f"Enqueued: {warm_start} trials for warm start")
        return warm_start
        
    
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
                    self.tpe_logs[trial.number]['params'][id] = selected
            logger.info(f"innerloop - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}") 
            
            states, score, price = evaluator(candidate)
            self.tpe_logs[trial.number]['score'] = score
            self.tpe_logs[trial.number]['price'] = price
            self.opt_cost += price
            logger.info(f"innerloop - Trial {trial.number} result: score: {score}, price: {price}")
            
            if self.fields_in_interest is not None:
                self.tpe_logs[trial.number]['fields'] = [state.all_news(self.fields_in_interest) for state in states]
            return score, price 
        return objective_function
        
    def optimize(
        self,
        workflow: Workflow,
        evaluator: Evaluator,
        n_trials: int,
        log_dir: str = 'holm_log',
    ):
        """Find optimal params for the given workflow
        
        Will not modify the given workflow
        """
        self.opt_cost = 0
        self.inner_loop_log_dir = os.path.join(log_dir, 'inner_loop')
        if not os.path.exists(self.inner_loop_log_dir):
            os.makedirs(self.inner_loop_log_dir, exist_ok=True)
        
        workflow.reset()
        tpe_log_path = os.path.join(self.inner_loop_log_dir, 'tpe_logs.json')
        self.prepare_params(workflow.get_all_modules(lambda x: isinstance(x, LangChainLM)))
        obj_func = self.get_objective_function(evaluator=evaluator, workflow=workflow)
        study = optuna.create_study(directions=[self.opt_direction, 'minimize']) # minimize for price
        
        if os.path.exists(tpe_log_path):
            with open(tpe_log_path, 'r') as f:
                self.tpe_logs = json.load(f)
                for trial_id, meta in self.tpe_logs.items():
                    trial = optuna.trial.create_trial(
                        params=meta['params'],
                        values=[meta['score'], meta['price']],
                        distributions=self.tpe_distributions,
                    )
                    study.add_trial(trial)
                    self.opt_cost += meta['price']
        else:
            warm_start = self.reduce_cold_start(study)
            study.optimize(obj_func, n_trials=n_trials)
            json.dump(self.tpe_logs, open(tpe_log_path, 'w+'), indent=4)
            
        for i, trial in enumerate(study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            mapped_params = {key: self.param_hash_2_options[key][idx] for key, idx in trial.params.items()}
            print("  Params: {}".format(mapped_params))
            f1, f2 = trial.values
            print("  Values: f1= {}, f2= {}".format(f1, f2))
        print("Opt Cost: {}".format(self.opt_cost))
        return study.best_trials