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

logger = logging.getLogger(__name__)

class OptimizationLayer:
    def __init__(
        self,
        name: str,
        evaluator: GeneralEvaluatorInterface,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 0,
    ):
        """
        The optimization will always try to minimize the price and maximize the score
        Please make sure the evaluator is consistent with this.
        
        Args:
            name: name of the optimization layer
            
            evaluator: the evaluator that will be used to evaluate the proposal
            
            dedicate_params: a list of params that is dedicated to pre-defined modules
                need to set `module_name` correctly
                
            universal_params: a list of params that will be broadcasted to all modules
                will ignore `module_name` field
            
            target_modules: if provided, only the modules in this list will be optimized
                this has higher priority than dedicated params
            
            save_ckpt_interval: if > 0, will save the optimization state every interval
                currently will overwrite the same file
        """
        self.name = name
        self.evaluator = evaluator
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
        
        self.opt_logs: dict[int, TrialLog] = None
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
        self.opt_target_lm_names: set[str] = None
        self.save_ckpt_interval = save_ckpt_interval
        
        self.top_down_info: TopDownInformation = None
    
    def prepare_opt_env(self):
        self.params = defaultdict(list)
        
        # NOTE: if param file exists, will load params from file and ignore the given params
        param_save_path = self.top_down_info.opt_config.param_save_path
        if os.path.exists(param_save_path):
            logger.info(f'Loading innerloop params from {param_save_path}')
            l_param = load_params(param_save_path)
            for param in l_param:
                self.params[param.module_name].append(param)
            allowed_lm_names = set(param.module_name for param in l_param)
        else:
            module_pool = self.top_down_info.current_module_pool
            # extract mappings from old_lm_name to all_new_lm_names
            old_2_new_lms: dict[str, list[str]] = defaultdict(list)
            allowed_lm_names = set()
            for new_module in module_pool.values():
                old_name, new_modules = self.top_down_info.module_ttrace.get_derivatives_of_same_type(new_module)
                new_names = [x.name for x in new_modules]
                if self.target_modules and old_name not in self.target_modules:
                    continue
                old_2_new_lms[old_name].extend(new_names)
                allowed_lm_names.update(new_names)
                
            # broadcast universal params
            if self.universal_params:
                for lm_name in allowed_lm_names:
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
                    elif param.module_name in allowed_lm_names:
                        target_names = [param.module_name]
                        
                    for lm_name in target_names:
                        mapped_param = copy.deepcopy(param)
                        mapped_param.module_name = lm_name
                        self.params[lm_name].append(mapped_param)
        
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        self.opt_target_lm_names = allowed_lm_names
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
        trial_params: dict[str, Any],
        program_copy: list[Module],
    ) -> tuple[list[Module], ModuleTransformTrace]:
        trace_for_next_level = copy.deepcopy(self.top_down_info.module_ttrace)
        opt_target_lms = Module.all_with_predicate(
            program_copy, lambda m: m.name in self.opt_target_lm_names
        )
        module_dict = {lm.name: lm for lm in opt_target_lms}
        new_modules = []
        changed_modules = set()
        for lm_name, params in self.params.items():
            for param in params:
                selected = trial_params[param.hash]
                new_module, new_mapping = param.apply_option(selected, module_dict[lm_name])
                for old_name, new_name in new_mapping.items():
                    trace_for_next_level.add_mapping(old_name, [new_name])
                new_modules.append(new_module)
                changed_modules.add(lm_name)
                trace_for_next_level.register_proposal(self.name, [(lm_name, param.name, selected)])
        
        for m_name, new_module in module_dict.items():
            if m_name not in changed_modules:
                new_modules.append(new_module)
        return new_modules, trace_for_next_level

    def propose(
        self,
        ms: list[Module],
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, list[Module], ModuleTransformTrace, str]]:
        """Propse and apply the next set of params
        
        Will return new set of modules without modifying the given modules
        """
        next_to_run = []
        for i in range(n_sample):
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            
            logger.info(f"- {self.name} - apply param - Trial {trial.number} params: {trial.params}")
            trial_log = TrialLog(params=trial.params, bo_trial_id=trial.number)
            self.opt_logs[trial_log.id] = trial_log
            program_copy = copy.deepcopy(ms)
            new_modules, new_trace = self._apply_params(trial.params, program_copy)
            next_to_run.append((trial, new_modules, new_trace, trial_log.id))
        return next_to_run

    def evaluate(
        self,
        new_top_down_info: TopDownInformation,
    ) -> EvaluationResult:
        eval_task = EvalTask.from_top_down_info(new_top_down_info)
        eval_result: EvaluationResult = self.evaluator.evaluate(eval_task)
        return eval_result
    
    def update(
        self,
        trial: optuna.trial.Trial,
        eval_result: EvaluationResult,
        log_id: str,
    ):
        score, price = eval_result.reduced_score, eval_result.reduced_price
        logger.info(f"- {self.name} - Trial {trial.number} result: score: {score}, price: {price}")
        self.opt_logs[log_id].score = score
        self.opt_logs[log_id].price = price
        
        total_price = sum(eval_result.prices)
        self.opt_logs[log_id].eval_cost = total_price
        self.opt_cost += total_price
         
        # update study if any dynamic params can evolve
        with self._study_lock:
            frozen_trial = self.study.tell(trial, [score, price])
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
    
    def save_ckpt(self, opt_log_path: str, param_save_path: str):
        opt_logs_json_obj = {k: v.to_dict() for k, v in self.opt_logs.items()}
        json.dump(opt_logs_json_obj, open(opt_log_path, 'w+'), indent=4)
        params = [param for params in self.params.values() for param in params]
        dump_params(params, param_save_path)
    
    def _optimize_iteration(
        self,
        base_program: list[Module],
    ):
        next_trial, program, new_trace, log_id = self.propose(base_program, 1)[0]
        
        # create info for next level optimization or actual evaluation
        next_level_info: TopDownInformation = copy.copy(self.top_down_info)
        next_level_info.module_ttrace = new_trace
        next_level_info.current_module_pool = {m.name: m for m in program}
        
        for lm_name, params in self.params.items():
            # NOTE: params might be updated when scheduling the current iteration
            # so we make a copy of the current params
            for param in params:
                next_level_info.all_params[param.hash] = copy.deepcopy(param)
        
        python_paths = next_level_info.other_python_paths + self._get_new_python_paths()
        python_paths = list(set(python_paths))
        next_level_info.other_python_paths = python_paths
        
        eval_result = self.evaluate(next_level_info)
        self.update(next_trial, eval_result, log_id)
    
    def _optimize(self, base_program: list[Module]):
        opt_config = self.top_down_info.opt_config
        if opt_config.throughput == 1:
            for _ in range(opt_config.n_trials):
                self._optimize_iteration(base_program)
        else:
            futures: set[Future] = set()
            counter = 0
            with ThreadPoolExecutor(max_workers=opt_config.throughput) as executor:
                for n_submitted_trials in range(opt_config.n_trials):
                    if len(futures) >= opt_config.throughput:
                        try:
                            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                            for f in completed:
                                try:
                                    f.result()
                                    counter += 1
                                    if self.save_ckpt_interval > 0 and counter % self.save_ckpt_interval == 0:
                                        self.save_ckpt(opt_config.opt_log_path, opt_config.param_save_path)
                                except Exception as e:
                                    logger.error(f'Error in evaluating task: {e}')
                        except Exception as e:
                            logger.error(f'Error in waiting for futures: {e}')
                    futures.add(executor.submit(self._optimize_iteration, base_program))
                wait(futures, return_when="ALL_COMPLETED")
                
    def get_pareto_front(self) -> list[TrialLog]:
        """
        Find the pareto-efficient points
        """
        trial_logs = []
        score_cost_list = []
        for log_id, trial_log in self.opt_logs.items():
            trial_logs.append(trial_log)
            score_cost_list.append((trial_log.score, trial_log.price))
        
        vectors = np.array([
            [-score, price] 
            for score, price in score_cost_list]
        )
        is_efficient = np.ones(vectors.shape[0], dtype = bool)
        for i, v in enumerate(vectors):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(vectors[is_efficient]<v, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        
        # return filtered [T_ParetoProgram]
        pareto_frontier = [log for log, eff in zip(trial_logs, is_efficient) if eff]
        return pareto_frontier
    
    def optimize(
        self,
        current_tdi: TopDownInformation,
    ):
        self.opt_cost = 0
        
        # prepare optimization environment
        current_tdi.initialize()
        self.top_down_info = current_tdi
        self.prepare_opt_env()
        
        # load previous optimization logs if exists
        opt_log_path = self.top_down_info.opt_config.opt_log_path
        if os.path.exists(opt_log_path):
            with open(opt_log_path, 'r') as f:
                opt_trace = json.load(f)
                
            for trial_log_id, trial_meta in opt_trace.items():
                trial_log = TrialLog.from_dict(trial_meta)
                self.opt_logs[trial_log_id] = trial_log
                self.opt_cost += trial_log.eval_cost
                
                trial = optuna.trial.create_trial(
                    params=trial_log.params,
                    values=[trial_log.score, trial_log.price],
                    distributions=self.param_categorical_dist,
                )
                self.study.add_trial(trial)
        
        # start optimization
        total_budget = self.top_down_info.opt_config.n_trials
        if total_budget > 0:
            logger.info(f"Start optimization {self.name} with {total_budget} trials")
            self._optimize(list(current_tdi.current_module_pool.values()))
            logger.info(f"Optimization {self.name} finished")
            self.save_ckpt(self.top_down_info.opt_config.opt_log_path,
                           self.top_down_info.opt_config.param_save_path)
        
        # Analysis optimization result
        pareto_frontier = self.get_pareto_front() 
        logger.info(f"--------- {self.name} Optimization Results ---------")
        for i, trial_log in enumerate(pareto_frontier):
            logger.info("The {}-th Pareto solution was found at Trial#{}.".format(i, trial_log.bo_trial_id))
            logger.info("  Params: {}".format(trial_log.params))
            logger.info("  Values: score= {}, price@1= {}".format(trial_log.score, trial_log.price))
            
        logger.info("Opt Cost: {}".format(self.opt_cost))
        logger.info(f"#Pareto Frontier: {len(pareto_frontier)}")
        logger.info("-------------------------------------------------")
        return self.opt_cost, pareto_frontier, self.opt_logs

    def easy_optimize(
        self, 
        opt_config: OptConfig,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ):
        tdi = TopDownInformation(
            opt_config=opt_config,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        return self.optimize(tdi)

    def easy_eval(
        self,
        trial_log_id: str,
        opt_config: OptConfig,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> EvaluationResult:
        # dummy optimization, populate all the necessary info
        opt_config.n_trials = 0
        cost, pareto_frontier, opt_logs = self.easy_optimize(
            opt_config=opt_config,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        
        # apply selected trial
        trial_log = self.opt_logs[trial_log_id]
        logger.info(f"----- Testing select trial {trial_log_id} -----")
        logger.info("  Params: {}".format(trial_log.params))
        logger.info("  Values: score= {}, price@1= {}".format(trial_log.score, trial_log.price))
        
        program_copy = copy.deepcopy(list(self.top_down_info.current_module_pool.values()))
        new_modules, new_trace = self._apply_params(
            trial_log.params, program_copy
        )
        next_level_info: TopDownInformation = copy.copy(self.top_down_info)
        next_level_info.module_ttrace = new_trace
        next_level_info.current_module_pool = {m.name: m for m in new_modules}
        
        for lm_name, params in self.params.items():
            # NOTE: params might be updated when scheduling the current iteration
            # so we make a copy of the current params
            for param in params:
                next_level_info.all_params[param.hash] = copy.deepcopy(param)
        
        python_paths = next_level_info.other_python_paths + self._get_new_python_paths()
        python_paths = list(set(python_paths))
        next_level_info.other_python_paths = python_paths
        
        # run evaluation
        eval_result = self.evaluate(next_level_info)
        return eval_result


class BottomLevelOptimization(OptimizationLayer):
    def __init__(
        self,
        name: str,
        evaluator: EvaluatorPlugin,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 0,
    ):
        super().__init__(
            name, evaluator, dedicate_params, universal_params, target_modules, save_ckpt_interval
        )
    