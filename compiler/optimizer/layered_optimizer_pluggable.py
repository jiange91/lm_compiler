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
from compiler.optimizer.params.common import ParamBase, OptionBase, DynamicParamBase, EvolveType, T_ModuleMapping, mmerge, mflatten, AddNewModuleImportInterface
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
        self.params: dict[str, list[ParamBase]] = None
        self.param_categorical_dist: dict[str, optuna.distributions.CategoricalDistribution] = None
        self.opt_logs: dict = None
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
        self.opt_target_lm_names: set[str] = None
    
    def prepare_opt_env(
        self, 
        module_pool: dict[str, Module], 
        flatten_module_mapping: T_ModuleMapping
    ):
        self.params = defaultdict(list)
        
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
        
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        self.opt_target_lm_names = all_opt_lm_names
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

    def _apply_params(
        self,
        trial: optuna.trial.Trial,
        program_copy: list[Module],
    ) -> Tuple[list[Module], T_ModuleMapping]:
        mapping: T_ModuleMapping = {}
        opt_target_lms = Module.all_with_predicate(
            program_copy, lambda m: m.name in self.opt_target_lm_names
        )
        module_dict = {lm.name: lm for lm in opt_target_lms}
        
        # NOTE: passed in ms will be used to replace the original ms
        # if optimize target is sub-module of a m, then m will be updated in-place
        # otherwise m itself need to be replaced by the optimize target
        optimize_itself = set([m.name for m in program_copy if m.name in module_dict])
        new_modules = [m for m in program_copy if m.name not in optimize_itself]
        self.opt_logs[trial.number] = {'params': {}}
        
        for lm_name, params in self.params.items():
            for param in params:
                selected = trial.params[param.hash]
                new_module, new_mapping = param.apply_option(selected, module_dict[lm_name])
                self.opt_logs[trial.number]['params'][param.hash] = selected
                mapping.update(new_mapping)
                if lm_name in optimize_itself:
                    module_dict[lm_name] = new_module
        
        for lm_name, new_module in module_dict.items():
            if lm_name in optimize_itself:
                new_modules.append(new_module)
        logger.info(f"- InnerLoop - next_to_run - Trial {trial.number} params: {self.opt_logs[trial.number]['params']}")
        return new_modules, mapping

    def propose(
        self,
        ms: list[Module],
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, list[Module], T_ModuleMapping]]:
        """Propse and apply the next set of params
        
        Will return new set of modules without modifying the given modules
        """
        next_to_run = []
        for i in range(n_sample):
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            
            # NOTE: this is the resulting program that should be used to replace original ms 
            program_copy = copy.deepcopy(ms)
            new_modules, mapping = self._apply_params(trial, program_copy)
            next_to_run.append((trial, new_modules, mapping))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        evaluator: EvaluatorPlugin,
        task: EvalTask,
    ):
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
    
    def _get_new_python_paths(self):
        new_python_paths = []
        for lm_name, params in self.params.items():
            for param in params:
                if isinstance(param, AddNewModuleImportInterface):
                    new_python_paths.extend(param.get_python_paths())
        return new_python_paths
    
    def _optimize(
        self,
        n_trials: int,
        script_path: str,
        other_python_paths: list[str],
        script_args: list[str],
        flatten_module_mapping: T_ModuleMapping,
        current_modules: list[Module],
        evaluator: EvaluatorPlugin,
        throughput: int = 1, # number of trials to run in parallel
    ):

        def _opt_loop(budget):
            for i in range(budget):
                next_trial, new_modules, new_mapping = self.propose(ms=current_modules, n_sample=1)[0]
                new_mapping = flatten_module_mapping | new_mapping
                new_mapping = mflatten(new_mapping)
                module_pool = {lm.name: lm for lm in new_modules}
                
                # register new module paths
                python_paths = other_python_paths + self._get_new_python_paths()
                # remove duplicate paths
                python_paths = list(set(python_paths))
                
                task = EvalTask(
                    script_path=script_path,
                    args=script_args,
                    module_map_table=new_mapping,
                    module_pool=module_pool,
                    other_python_paths=python_paths,
                )
                self._eval_and_update(
                    trial=next_trial,
                    evaluator=evaluator,
                    task=task,
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
        other_python_paths: Optional[list[str]] = None,
    ) -> Tuple[float, list[Tuple[optuna.trial.FrozenTrial, EvalTask]]]:
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
            logger.info(f'Loading innerloop params from {param_save_path}')
            params = load_params(param_save_path)
            self.dedicated_params = params
            self.universal_params = []
        
        script_args = script_args or []
        other_python_paths = other_python_paths or []
        if module_pool is None:
            dir = os.path.dirname(script_path)
            if dir not in sys.path:
                sys.path.insert(0, dir)
            sys.argv = [script_path] + script_args
            schema = OptimizerSchema.capture(script_path)
            module_pool = {m.name: m for m in schema.opt_target_modules}
            
        for m in module_pool.values():
            m.enclosing_module = None # this is to prevent pickling enclosing module
            m.reset()
        
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
                other_python_paths=other_python_paths,
                script_args=script_args,
                flatten_module_mapping=flatten_module_mapping,
                current_modules=list(module_pool.values()),
                evaluator=evaluator,
                throughput=throughput,
            )
            json.dump(self.opt_logs, open(opt_log_path, 'w+'), indent=4)
            params = [param for params in self.params.values() for param in params]
            dump_params(params, param_save_path)
         
        pareto_frontier = []
        for i, trial in enumerate(self.study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: score= {}, cost= {}".format(f1, f2))
            
            # cache optimized program
            program_cpy = copy.deepcopy(list(module_pool.values()))
            new_modules, new_mapping = self._apply_params(trial, program_cpy)
            new_module_pool = {m.name: m for m in new_modules}
            new_mapping = mflatten(new_mapping | flatten_module_mapping)
            
            python_paths = other_python_paths + self._get_new_python_paths()
            python_paths = list(set(python_paths))
            task = EvalTask(
                script_path=script_path,
                args=script_args,
                module_map_table=new_mapping,
                module_pool=new_module_pool,
                other_python_paths=python_paths,
            )
            pareto_frontier.append((trial, task))
            
        print("Opt Cost: {}".format(self.opt_cost))
        return self.opt_cost, pareto_frontier

class InnerLoopEvaluator:
    def __init__(
        self,
        inner_loop: InnerLoopBayesianOptimization,
        n_trials: int,
        evaluator: EvaluatorPlugin,
        throughput: int = 1,
        log_dir: str = 'holm_log',
        quality_constraint: float = None,
    ):
        self.evaluator = evaluator
        self.inner_loop = inner_loop
        self.n_trials = n_trials
        self.throughput = throughput
        self.log_dir = log_dir
        self.quality_constraint = quality_constraint
    
    def __call__(
        self,
        task: EvalTask
    ) -> EvaluationResult:
        pareto_frontier, opt_cost = self.inner_loop.optimize(
            script_path=task.script_path,
            n_trials=self.n_trials,
            evaluator=self.evaluator,
            throughput=self.throughput,
            log_dir=self.log_dir,
            script_args=task.args,
            module_pool=task.module_pool,
            flatten_module_mapping=task.module_map_table,
            other_python_paths=task.other_python_paths,
        )
        best_scores, best_prices = [], []
        for trial in pareto_frontier:
            if self.quality_constraint is not None and trial.values[0] < self.quality_constraint:
                continue
            best_scores.append(trial.values[0])
            best_prices.append(trial.values[1])
        
        if len(best_scores) == 0:
            reduced_score, reduced_price = float('-inf'), float('inf')
        else:
            reduced_score, reduced_price = max(best_scores), min(best_prices)
        result = EvaluationResult(
            scores=best_scores,
            prices=best_prices,
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            demos=None,
            meta={'inner_opt_cost': opt_cost}
        )
        return result

class OuterLoopOptimization:
    
    def __init__(
        self,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        quality_constraint: float = None,
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
            
        """
        self.dedicated_params = dedicate_params
        self.universal_params = universal_params
        self.quality_constraint = quality_constraint
        if len(self.dedicated_params) + len(self.universal_params) == 0:
            raise ValueError('No params provided for optimization')
        
        self.opt_direction = 'maximize'
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.opt_cost = 0

        # will be updated when prepare_opt_env is called
        self.params: dict[str, list[ParamBase]] = None
        self.opt_base_lms: list[LangChainLM] = None
        self.param_categorical_dist: dict[str, optuna.distributions.CategoricalDistribution] = None
        self.opt_logs: dict = None
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
        self.best_score_cost = []
        
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
    
    def prepare_opt_env(
        self,
        lm_modules: list[LangChainLM],
    ):
        self.params = defaultdict(list)
        
        if self.universal_params:
            for lm in lm_modules:
                lm_name = lm.name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                params_cpy = copy.deepcopy(self.universal_params)
                for param in params_cpy:
                    param.module_name = lm_name
                self.params[lm_name] = params_cpy
                
        if self.dedicated_params:
            for param in self.dedicated_params:
                lm_name = param.module_name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                self.params[lm_name].append(param)
                
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }

        self.study: optuna.Study = self.init_study()
        self._study_lock = threading.Lock()
        self.opt_logs = {}
        self.best_score_cost = []
    
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
            new_modules = []
            for lm_name, params in self.params.items():
                assert len(params) == 1, 'Only one param per module is supported at outerloop'
                for param in params:
                    selected = trial.params[param.hash]
                    new_module, new_mapping = param.apply_option(selected, module_dict[lm_name])
                    self.opt_logs[trial.number]['params'][param.hash] = selected
                    mapping.update(new_mapping)
                    new_modules.append(new_module)
                    
            logger.info(f"- OuterLoop - next_to_run - Trial {trial.number} params: {self.opt_logs[trial.number]['params']}")
            next_to_run.append((trial, new_modules, mapping))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        task: EvalTask,
        inner_evaluator: InnerLoopEvaluator,
    ):
        eval_result: EvaluationResult = inner_evaluator(task)
        for p_score, p_price in zip(eval_result.scores, eval_result.prices):
            self.best_score_cost.append((p_score, p_price))
        inner_cost = eval_result.meta['inner_opt_cost']
        self.opt_cost += inner_cost
        
        score, price = eval_result.reduced_score, eval_result.reduced_price
        logger.info(f"- OuerLoop - Trial {trial.number} result: score: {score}, price: {price}")
        self.opt_logs[trial.number]['score'] = score
        self.opt_logs[trial.number]['price'] = price
        self.opt_logs[trial.number]['total_price'] = inner_cost
         
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
                
    def _get_new_python_paths(self):
        new_python_paths = []
        for lm_name, params in self.params.items():
            for param in params:
                if isinstance(param, AddNewModuleImportInterface):
                    new_python_paths.extend(param.get_python_paths())
        return new_python_paths
                
    def _opt_loop(
        self,
        n_outer_trials,
        n_inner_trials,
        script_path: str,
        script_args: list[str],
        lms: list[LangChainLM],
        evaluator: EvaluatorPlugin,
        inner_loop: InnerLoopBayesianOptimization,
        inner_throughput: int,
        log_dir: str,
    ):
        for i in range(n_outer_trials):
            next_trial, program, mapping = self.propose(lms, 1)[0]
            
            # use inner loop optimization as evaluator
            mapping = mflatten(mapping)
            
            # register new module paths
            python_paths = self._get_new_python_paths()
            # remove duplicate paths
            python_paths = list(set(python_paths))
            
            eval_task = EvalTask(
                script_path=script_path,
                args=script_args,
                module_map_table=mapping,
                module_pool={lm.name: lm for lm in program},
                other_python_paths=python_paths,
            )
            feedback_consumer = InnerLoopEvaluator(
                inner_loop=copy.deepcopy(inner_loop),
                n_trials=n_inner_trials,
                evaluator=evaluator,
                throughput=inner_throughput,
                log_dir=os.path.join(log_dir, 'inner_loop', uuid.uuid4().hex),
                quality_constraint=self.quality_constraint,
            )
            self._eval_and_update(next_trial, eval_task, feedback_consumer)
    
    def _optimize(
        self, 
        n_outer_trials: int,
        n_inner_trials: int,
        script_path: str,
        script_args: list[str],
        lms: list[LangChainLM],
        evaluator: EvaluatorPlugin,
        log_dir: str,
        inner_loop: InnerLoopBayesianOptimization,
        throughput: int, # number of trials to run in parallel
        inner_throughput: int,
    ):
        if throughput == 1:
            self._opt_loop(
                n_outer_trials=n_outer_trials,
                n_inner_trials=n_inner_trials,
                script_path=script_path,
                script_args=script_args,
                lms=lms,
                evaluator=evaluator,
                inner_loop=inner_loop,
                inner_throughput=inner_throughput,
                log_dir=log_dir,
            )
        else:
            futures: set[Future] = set()
            with ThreadPoolExecutor(max_workers=throughput) as executor:
                for n_submitted_trials in range(n_outer_trials):
                    if len(futures) >= throughput:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in completed:
                            f.result()
                    futures.add(
                        executor.submit(
                            self._opt_loop, 
                            1, 
                            n_outer_trials, n_inner_trials, script_path, script_args, lms, evaluator, inner_loop, inner_throughput, log_dir)
                        )
                    
    def get_pareto_front(self):
        """
        Find the pareto-efficient points
        """
        vectors = np.array([
            [-score, price] 
            for score, price in self.best_score_cost]
        )
        is_efficient = np.ones(vectors.shape[0], dtype = bool)
        for i, v in enumerate(vectors):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(vectors[is_efficient]<v, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        masked = np.array(self.best_score_cost)[is_efficient]
        return masked
    
    def optimize(
        self,
        inner_loop: InnerLoopBayesianOptimization,
        n_trials: int,
        script_path: str,
        evaluator: EvaluatorPlugin,
        script_args: list[str] = None,
        module_pool: dict[str, Module] = None,
        
        resource_ratio: float = 1 / 10, # trials in outer vs inner loop
        throughput: int = 1,
        inner_throughput: int = 1,
        log_dir: str = 'holm_log',
    ):
        self.opt_cost = 0
        self.outer_loop_log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        tpe_log_path = os.path.join(self.outer_loop_log_dir, 'outer_opt_logs.json')
        param_save_path = os.path.join(self.outer_loop_log_dir, 'outer_params.json')
        
        if os.path.exists(param_save_path):
            logger.info(f'Loading outerloop params from {param_save_path}')
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
            lm_modules = schema.opt_target_modules
            
        for lm in lm_modules:
            lm.enclosing_module = None
            lm.reset()
            
        self.prepare_opt_env(lm_modules)
        
        if os.path.exists(tpe_log_path):
            with open(tpe_log_path, 'r') as f:
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
            n_outer_trials = math.ceil(n_trials * resource_ratio)
            n_inner_trials = min(n_trials, math.floor(1 / resource_ratio))
            self._optimize(
                n_outer_trials=n_outer_trials,
                n_inner_trials=n_inner_trials,
                script_path=script_path,
                script_args=script_args,
                lms=lm_modules,
                evaluator=evaluator,
                log_dir=log_dir,
                inner_loop=inner_loop,
                throughput=throughput,
                inner_throughput=inner_throughput,
            )
            json.dump(self.opt_logs, open(tpe_log_path, 'w+'), indent=4)
            params = [param for params in self.params.values() for param in params]
            dump_params(params, param_save_path)
            
        for i, trial in enumerate(self.study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: best score= {}, eff cost= {}".format(f1, f2))
        print("Total Opt Cost: {}".format(self.opt_cost))
        
        pareto_paris = self.get_pareto_front()
        for i, (score, price) in enumerate(pareto_paris):
            print(f"Pareto Point {i}: score={score}, price={price}")
        return self.study.best_trials, self.opt_cost
    
    