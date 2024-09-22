import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal
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
from dataclasses import dataclass
import multiprocessing as mp

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase, DynamicParamBase, EvolveType, T_ModuleMapping, mmerge, mflatten
from compiler.optimizer.params.utils import dump_params, load_params
from compiler.optimizer.params.model_selection import LMSelection
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from compiler.optimizer.plugin import OptimizerSchema
from optuna.samplers import TPESampler


logger = logging.getLogger(__name__)

    

class InnerLoopBayesianOptimization:
    def __init__(
        self,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
    ):
        """
        The optimization will always try to minimize the price and maximize the score
        Please make sure the evaluator is consistent with this.
        
        Args:
            d_params: a list of params that is dedicated to pre-defined modules
                need to set `module_name` correctly
                
            u_params: a list of params that will be broadcasted to all modules
                will ignore `module_name` field
            
            target_modules: if provided, only the modules in this list will be optimized
                this has higher priority than dedicated params
        """
        self.dedicated_params = dedicate_params
        self.universal_params = universal_params
        if len(self.dedicated_params) + len(self.universal_params) == 0:
            raise ValueError('No params provided for optimization')
        
        self.opt_direction = 'maximize'
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.opt_cost = 0

        # will be updated when prepare_opt_env is called
        self.opt_base_lms: list[LangChainLM] = None
        self.param_categorical_dist: dict[str, optuna.distributions.CategoricalDistribution] = None
        self.opt_logs: dict = None
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
    
    def prepare_opt_env(
        self, 
        module_pool: dict[str, Module], 
        flatten_module_mapping: T_ModuleMapping
    ):
        self.params: dict[str, list[ParamBase]] = defaultdict(list)
        
        # extract mappings from old_lm_name to all_new_lm_names
        # NOTE: module_pool can contain compositable modules
        old_2_new_lms: dict[str, list[str]] = {}
        allowed_new_lms = set()
        for old_lm, new_name in flatten_module_mapping.items():
            if self.target_modules and old_lm not in self.target_modules:
                continue
            new_lms = Module.all_of_type([module_pool[new_name]], LangChainLM)
            old_2_new_lms[old_lm] = [lm.name for lm in new_lms]
            allowed_new_lms.update(old_2_new_lms[old_lm])
            
        # get names of lm modules that will be optimized
        all_opt_lms = Module.all_of_type(module_pool.values(), LangChainLM)
        all_opt_lm_names = set([lm.name for lm in all_opt_lms])
        if self.target_modules:
            allowed_lm_names = set(self.target_modules) | allowed_new_lms
            all_opt_lm_names = all_opt_lm_names & allowed_lm_names
         
        # broadcast universal params
        if self.universal_params:
            for lm_name in all_opt_lm_names:
                params_cpy = copy.deepcopy(self.universal_params)
                for param in params_cpy:
                    param.module_name = lm_name
                self.params[lm_name] = params_cpy
        
        # apply dedicated params
        if self.dedicated_params:
            for param in self.dedicated_params:
                target_names = []
                if param.module_name in old_2_new_lms:
                    target_names = old_2_new_lms[param.module_name]
                elif param.module_name in all_opt_lm_names:
                    target_names = [param.module_name]
                    
                for lm_name in target_names:
                    mapped_param = copy.deepcopy(param)
                    mapped_param.module_name = lm_name
                    self.params[lm_name].append(mapped_param)
        
        self.opt_base_lms = all_opt_lms
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        self.opt_logs = {}
        self.study = self.init_study()
        self._study_lock = threading.Lock()

    def init_study(
        self,
        old_study: optuna.Study = None,
        old_trials: list[optuna.trial.Trial] = None,
    ):
        """Create a new study and migrate old trials if provided
        
        For all provided trials, the params dist will be adjusted to the current
        self.params when this method is called.
        
        Recommand using name based options instead of index based options as the dynamic
        params update may change the mapping between option index and the option itself
        """
        new_study = optuna.create_study(
            directions=['maximize', 'minimize'],
            sampler=TPESampler(multivariate=True)
        )
        
        f_trials = []
        if old_study:
            f_trials.extend(old_study.trials)
        if old_trials:
            f_trials.extend(old_trials)    
        
        for trial in f_trials:
            # Modify previous trials. 
            params = trial.params
            dists = self.param_categorical_dist
            # These changes are not persisted to the storage.
            trial.params = params
            trial.distributions = dists
            # Persist the changes to the storage (in a new study).
            new_study.add_trial(trial)
        return new_study

    def propose(
        self,
        lms: list[LangChainLM],
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, list[LangChainLM], T_ModuleMapping]]:
        """Propse and apply the next set of params
        
        Will return new set of modules without modifying the given modules
        """
        next_to_run = []
        for i in range(n_sample):
            mapping: T_ModuleMapping = {}
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            lms_copy = copy.deepcopy(lms)
            module_dict = {lm.name: lm for lm in lms_copy}
            self.opt_logs[trial.number] = {'params': {}}
            for lm_name, params in self.params.items():
                for param in params:
                    selected = trial.params[param.hash]
                    new_module, new_mapping = param.apply_option(selected, module_dict[lm_name])
                    self.opt_logs[trial.number]['params'][param.hash] = selected
                    mapping.update(new_mapping)
                    
            logger.info(f"- InnerLoop - next_to_run - Trial {trial.number} params: {self.opt_logs[trial.number]['params']}")
            next_to_run.append((trial, lms_copy, mapping))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        script_path: str,
        script_args: list[str],
        flatten_module_mapping: T_ModuleMapping,
        new_modules: list[Module],
        evaluator: EvaluatorPlugin,
    ):
        module_pool = {lm.name: lm for lm in new_modules}
        task = EvalTask(
            script_path=script_path,
            args=script_args,
            module_map_table=flatten_module_mapping,
            module_pool=module_pool,
        )
        eval_result: EvaluationResult = evaluator.evaluate(task)
        score, price = eval_result.reduced_score, eval_result.reduced_price
        logger.info(f"- Trial {trial.number} result: score: {score}, price: {price}")
        self.opt_logs[trial.number]['score'] = score
        self.opt_logs[trial.number]['price'] = price
        
        total_price = sum(eval_result.prices)
        self.opt_logs[trial.number]['total_price'] = total_price
        self.opt_cost += total_price
         
        # update study if any dynamic params can evolve
        with self._study_lock:
            self.study.tell(trial, [score, price])
            is_evolved = False
            for params in self.params.values():
                for param in params:
                    if isinstance(param, DynamicParamBase):
                        evolve_type = param.evole(eval_result)
                        if evolve_type != EvolveType.ID:
                            is_evolved = True
            if is_evolved:
                # update param dist
                self.param_categorical_dist = {
                    param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
                    for _, params in self.params.items() for param in params
                }
                # create new study and migrate all trials
                new_study = self.init_study(self.study)
                self.study = new_study
    
    def _optimize(
        self,
        n_trials: int,
        script_path: str,
        script_args: list[str],
        flatten_module_mapping: T_ModuleMapping,
        current_modules: list[LangChainLM],
        evaluator: EvaluatorPlugin,
        throughput: int = 1, # number of trials to run in parallel
    ):

        def _opt_loop(budget):
            for i in range(budget):
                next_trial, new_modules, new_mapping = self.propose(lms=current_modules, n_sample=1)[0]
                new_mapping.update(flatten_module_mapping)
                new_mapping = mflatten(new_mapping)
                self._eval_and_update(
                    trial=next_trial,
                    script_path=script_path,
                    script_args=script_args,
                    flatten_module_mapping=new_mapping,
                    new_modules=new_modules,
                    evaluator=evaluator,
                )
                
        if throughput == 1:
            _opt_loop(n_trials)
        else:
            futures: set[Future] = set()
            with ThreadPoolExecutor(max_workers=throughput) as executor:
                for n_submitted_trials in range(n_trials):
                    if len(futures) >= throughput:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in completed:
                            f.result()
                    futures.add(executor.submit(_opt_loop, 1))
        
    def optimize(
        self,
        script_path: str,
        n_trials: int,
        evaluator: EvaluatorPlugin,
        throughput: int = 1,
        log_dir: str = 'holm_log',
        script_args: list[str] = None,
        module_pool: dict[str, Module] = None,
        flatten_module_mapping: T_ModuleMapping = None,
    ) -> list[optuna.trial.FrozenTrial]:
        """Find optimal params for the given workflow
        
        Args:
            script_path: path to the script that will be executed
            
            module_mapping: mapping between old module names and new module names
                this does not contain modules that are not replaced
            
            module_pool: modules that are available for the optimization
                including new ones to be replaced
        """
        self.opt_cost = 0
        self.inner_loop_log_dir = log_dir
        if not os.path.exists(self.inner_loop_log_dir):
            os.makedirs(self.inner_loop_log_dir, exist_ok=True)
        
        opt_log_path = os.path.join(self.inner_loop_log_dir, 'opt_logs.json')
        param_save_path = os.path.join(self.inner_loop_log_dir, 'params.json')
        
        # NOTE: if param file exists, will load params from file and ignore the given params
        if os.path.exists(param_save_path):
            logger.info(f'Loading params from {param_save_path}')
            params = load_params(param_save_path)
            self.dedicated_params = params
            self.universal_params = []
        
        script_args = script_args or []
        if module_pool is None:
            dir = os.path.dirname(script_path)
            if dir not in sys.path:
                sys.path.insert(0, dir)
            sys.argv = [script_path] + script_args
            schema = OptimizerSchema.capture(script_path)
            module_pool = {m.name: m for m in schema.opt_target_modules}
            
        for lm in module_pool.values():
            lm.reset()
        
        flatten_module_mapping = flatten_module_mapping or {}
        self.prepare_opt_env(module_pool, flatten_module_mapping)
        
        if os.path.exists(opt_log_path):
            with open(opt_log_path, 'r') as f:
                self.opt_logs = json.load(f)
                for trial_id, meta in self.opt_logs.items():
                    trial = optuna.trial.create_trial(
                        params=meta['params'],
                        values=[meta['score'], meta['price']],
                        distributions=self.param_categorical_dist,
                    )
                    self.study.add_trial(trial)
                    self.opt_cost += meta['total_price']
        else:
            self._optimize(
                n_trials=n_trials,
                script_path=script_path,
                script_args=script_args,
                flatten_module_mapping=flatten_module_mapping,
                current_modules=self.opt_base_lms,
                evaluator=evaluator,
                throughput=throughput,
            )
            json.dump(self.opt_logs, open(opt_log_path, 'w+'), indent=4)
            params = [param for params in self.params.values() for param in params]
            dump_params(params, param_save_path)
         
        for i, trial in enumerate(self.study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: score= {}, cost= {}".format(f1, f2))
        print("Opt Cost: {}".format(self.opt_cost))
        return self.study.best_trials, self.opt_cost
    