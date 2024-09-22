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
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
import math
import threading
import uuid

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase, DynamicParamBase, EvolveType, T_ModuleMapping, merge_module_mapping, flatten_mapping
from compiler.optimizer.params.utils import dump_params, load_params
from compiler.optimizer.params.model_selection import LMSelection
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from optuna.samplers import TPESampler


logger = logging.getLogger(__name__)

class OptRoutineBase(ABC):
    
    def __init__(
        self,
        workflow: Workflow,
        opt_directions: list[str],
        params: dict[str, list[ParamBase]],
        quality_constraint: Optional[float] = None,
    ):
        self.workflow = workflow
        self.params = params
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        self.quality_constraint = quality_constraint
        
        self.study = self.init_study(
            opt_directions=opt_directions,
            old_study=None,
            old_trials=None,
        )
        self._study_lock = threading.Lock()
        self.opt_logs = {}
    
    def init_study(
        self,
        opt_directions: Optional[list[str]] = None,
        old_study: optuna.Study = None,
        old_trials: list[optuna.trial.Trial] = None,
    ):
        """Create a new study and migrate old trials if provided
        
        For all provided trials, the params dist will be adjusted to the current
        self.params when this method is called.
        
        Recommand using name based options instead of index based options as the dynamic
        params update may change the mapping between option index and the option itself
        """
        if old_study:
            opt_directions = old_study.directions
        if not opt_directions:
            raise ValueError('opt_directions is required')
        
        new_study = optuna.create_study(
            directions=opt_directions,
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
    
    @abstractmethod
    def propose(self, n_sample) -> list[Tuple[optuna.trial.Trial, Workflow]]:
        """Based on existing observations, propose new configs
        """
        pass
    
    @abstractmethod
    def evaluate(self, trial: optuna.trial.Trial, workflow: Workflow, evaluator: EvaluatorInterface):
        pass
        
    
    @abstractmethod
    def update(self, feedback):
        """Consume feedbacks from rear layers and update search strategy
        """
        pass

class LLMOptRoutine(OptRoutineBase):
    
    def propose(self, n_sample) -> list[Tuple[optuna.trial.Trial, Workflow]]:
        next_to_run = []
        for i in range(n_sample):
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            program = copy.deepcopy(self.workflow)
            module_dict = {
                lm.name: lm 
                for lm in program.get_all_modules(lambda x: isinstance(x, LangChainLM))
            }
            self.opt_logs[trial.number] = {'params': {}}
            for lm_name, params in self.params.items():
                for param in params:
                    selected = trial.params[param.hash]
                    param.apply_option(selected, module_dict[lm_name])
                    self.opt_logs[trial.number]['params'][param.hash] = selected
                    
            logger.info(f"next_to_run - Trial {trial.number} params: {self.opt_logs[trial.number]['params']}")
            next_to_run.append((trial, program))
        return next_to_run
    
    def evaluate(
        self, 
        trial: optuna.trial.Trial, 
        workflow: Workflow, 
        evaluator: EvaluatorInterface
    ):
        eval_result: EvaluationResult = evaluator(workflow)
        pass

    def update(self, feedback):
        return self.update_regardless(feedback)
    
    def update_regardless(self, feedback: dict[int, list[Tuple[float, float]]]):
        for trial_number, score_price_list in feedback.items():
            if len(score_price_list) == 1:
                # we are at bottom layer
                self.study.tell(trial_number, score_price_list[0])
            else:
                """
                Each trial receives the Pareto front from the next layer as feedback
                To quantify the value of a frontier we use mean score and price as the indicator
                """
                mean_score = np.mean([score for score, _ in score_price_list])
                mean_price = np.mean([price for _, price in score_price_list])
                self.study.tell(trial_number, [mean_score, mean_price])
    
    def update_with_reference(self, feedback: dict[int, list[Tuple[float, float]]]):
        for trial_number, score_price_list in feedback.items():
            if len(score_price_list) == 1:
                # we are at bottom layer
                self.study.tell(trial_number, score_price_list[0])
            else:
                """
                Each trial receives the Pareto front from the next layer as feedback
                To quantify the value of a frontier we use (best_score, best_price_within_gap) as the indicate
                Can change to more advanced metrics e.g. hypervolume indicator
                """
                best_score, best_price_within_gap = score_price_list[0]
                for score, price in score_price_list:
                    best_score = max(best_score, score)
                    if self.quality_constraint(score):
                        best_price_within_gap = min(best_price_within_gap, price)
                self.study.tell(trial_number, [best_score, best_price_within_gap])
        
class LayerBase(ABC):
    """Building block of hierarchical optimization
    
    Each layer control the optimization process given a workflow
    
    It controls the resource allocation and the optimization strategy
    """
    def __init__(
        self,
        name: str,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
    ):
        """Create a LOPT instance with configs
        """
        self.name = name
        self.target_modules = set(target_modules) if target_modules is not None else {}
        self.dedicated_params = dedicate_params
        self.universal_params = universal_params
        
    @abstractmethod
    def prepare_optimization_routine(self, workflow: Workflow, opt_directions: list[str]) -> OptRoutineBase:
        pass


class GeneralLLMOptLayer(LayerBase):
    """General Layer for LLM cost-effective optimization
    """
    def __init__(
        self, 
        name: str,
        module_mapping: Optional[dict[str, list[str]]] = None,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        quality_constraint: float = None,
    ):
        """
        
        Args:
            module_mapping: 
                a dict that maps the old module name used in the params to the new module name in the workflow
                this is useful when previous layers have changed the module name
                
            quality_constraint: 
                if provided, the optimization will ignore the cost saving if the score is below this value
        """
        assert module_mapping is None, "module_mapping is not supported yet"
        super().__init__(
            name=name, 
            dedicate_params=dedicate_params, 
            universal_params=universal_params, 
            target_modules=target_modules
        )
        self.opt_logs = {}
        self.module_mapping = module_mapping
        self.quality_constraint = quality_constraint
    
    def prepare_optimization_routine(
        self, 
        workflow: Workflow,
        opt_directions: list[str],
    ) -> OptRoutineBase:
        lm_modules = workflow.get_all_modules(lambda x: isinstance(x, LangChainLM))
        _params: dict[str, list[ParamBase]] = {}
        
        if self.universal_params:
            for lm in lm_modules:
                lm_name = lm.name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                params_cpy = copy.deepcopy(self.universal_params)
                for param in params_cpy:
                    param.module_name = lm_name
                _params[lm_name] = params_cpy
                
        if self.dedicated_params:
            for param in self.dedicated_params:
                lm_name = param.module_name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                _params[lm_name].append(param)
        
        # take a snapshot of the program
        opt_routine = LLMOptRoutine(
            workflow=copy.deepcopy(workflow),
            opt_directions=opt_directions,
            params=_params,
            quality_constraint=self.quality_constraint,
        )
        return opt_routine
    

class InnerLoopBayesianOptimization:
    def __init__(
        self,
        dedicate_params: list[ParamBase] = [],
        universal_params: list[ParamBase] = [],
        target_modules: Iterable[str] = None,
        fields_in_interest: list[str] = None,
        important_lms: list[str] = None,
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
        if len(self.dedicated_params) + len(self.universal_params) == 0:
            raise ValueError('No params provided for optimization')
        
        self.opt_direction = 'maximize'
        self.fields_in_interest = fields_in_interest
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.important_lms = important_lms
        self.opt_cost = 0
    
    def prepare_params(self, lm_modules: list[LangChainLM], module_mapping: T_ModuleMapping):
        self.params: dict[str, list[ParamBase]] = defaultdict(list)
        current_lm_names = set([lm.name for lm in lm_modules])
        
        # broadcast universal params
        if self.target_modules:
            mapped_target_modules = set(self.target_modules)
            for old_lm, mapped in module_mapping.items():
                if old_lm in self.target_modules:
                    mapped_target_modules.update(mapped)
        else:
            mapped_target_modules = None
        if self.universal_params:
            for lm_name in current_lm_names:
                if mapped_target_modules and lm_name not in mapped_target_modules:
                    continue
                params_cpy = copy.deepcopy(self.universal_params)
                for param in params_cpy:
                    param.module_name = lm_name
                self.params[lm_name] = params_cpy
        
        # apply dedicated params
        if self.dedicated_params:
            for param in self.dedicated_params:
                if self.target_modules and param.module_name not in self.target_modules:
                    continue
                target_names = set([param.module_name])
                target_names.update(module_mapping.get(param.module_name, []))
                for lm_name in target_names:
                    if lm_name in current_lm_names:
                        mapped_param = copy.deepcopy(param)
                        mapped_param.module_name = lm_name
                        self.params[lm_name].append(mapped_param)
                
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        self.tpe_logs = {}
        self.study = None
        self._study_lock = threading.Lock()
    
    def reduce_cold_start(self, study: optuna.Study):
        # set base config
        base_config = {param.hash: param.default_option 
                       for _, params in self.params.items() 
                       for param in params}
        study.enqueue_trial(base_config)
        # create trials for important lms
        for lm_name, params in self.params.items():
            if lm_name in self.important_lms:
                for param in params:
                    if isinstance(param, LMSelection):
                        for option in param.options:
                            if option != param.default_option:
                                config = copy.deepcopy(base_config)
                                config[param.hash] = option
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
            
            logger.info(f"InnerLoop - start Trial: {trial.number}")
            self.tpe_logs[trial.number] = {}
            self.tpe_logs[trial.number]['params'] = {}
            
            for lm_name, params in self.params.items():
                for param in params:
                    id = param.hash
                    selected = trial.suggest_categorical(id, self.param_categorical_dist[id])
                    new_lm, mapping = param.apply_option(selected, module_dict[lm_name])
                    module_dict[lm_name] = new_lm
                    self.tpe_logs[trial.number]['params'][id] = selected
            logger.info(f"- InnerLoop - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}") 
            
            states, score, price = evaluator(candidate)
            self.tpe_logs[trial.number]['score'] = score
            self.tpe_logs[trial.number]['price'] = price
            self.opt_cost += price
            logger.info(f"- InnerLoop - Trial {trial.number} result: score: {score}, price: {price}")
            
            if self.fields_in_interest is not None:
                self.tpe_logs[trial.number]['fields'] = [state.all_news(self.fields_in_interest) for state in states]
            return score, price 
        return objective_function

    def propose(
        self,
        workflow: Workflow,
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, Workflow]]:
        next_to_run = []
        for i in range(n_sample):
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            program = copy.deepcopy(workflow)
            module_dict = {
                lm.name: lm 
                for lm in program.get_all_modules(lambda x: isinstance(x, LangChainLM))
            }
            self.tpe_logs[trial.number] = {'params': {}}
            for lm_name, params in self.params.items():
                for param in params:
                    selected = trial.params[param.hash]
                    param.apply_option(selected, module_dict[lm_name])
                    self.tpe_logs[trial.number]['params'][param.hash] = selected
                    
            logger.info(f"- InnerLoop - next_to_run - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}")
            program.compile()
            next_to_run.append((trial, program))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        workflow: Workflow,
        evaluator: EvaluatorInterface,
    ):
        eval_result: EvaluationResult = evaluator(workflow)
        score, price = eval_result.reduced_score, eval_result.reduced_price
        logger.info(f"- Trial {trial.number} result: score: {score}, price: {price}")
        self.tpe_logs[trial.number]['score'] = score
        self.tpe_logs[trial.number]['price'] = price
        
        total_price = sum(eval_result.prices)
        self.tpe_logs[trial.number]['total_price'] = total_price
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
    
    def _opt_loop(self, n_trials, workflow, evaluator):
        for i in range(n_trials):
            next_trial, program = self.propose(workflow, 1)[0]
            self._eval_and_update(next_trial, program, evaluator)
    
    def _optimize(
        self, 
        n_trials: int,
        workflow: Workflow, 
        evaluator: EvaluatorInterface,
        throughput: int = 1, # number of trials to run in parallel
    ):
        if throughput == 1:
            self._opt_loop(n_trials, workflow, evaluator)
        else:
            futures: set[Future] = set()
            with ThreadPoolExecutor(max_workers=throughput) as executor:
                for n_submitted_trials in range(n_trials):
                    if len(futures) >= throughput:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in completed:
                            f.result()
                    futures.add(executor.submit(self._opt_loop, 1, workflow, evaluator))
    
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
        
        
    def optimize(
        self,
        workflow: Workflow,
        module_mapping: T_ModuleMapping,
        evaluator: EvaluatorInterface,
        n_trials: int,
        throughput: int = 1,
        log_dir: str = 'holm_log',
    ) -> list[optuna.trial.FrozenTrial]:
        """Find optimal params for the given workflow
        
        Will not modify the given workflow
        """
        self.opt_cost = 0
        self.inner_loop_log_dir = log_dir
        if not os.path.exists(self.inner_loop_log_dir):
            os.makedirs(self.inner_loop_log_dir, exist_ok=True)
        
        workflow.reset()
        tpe_log_path = os.path.join(self.inner_loop_log_dir, 'tpe_logs.json')
        param_save_path = os.path.join(self.inner_loop_log_dir, 'params.json')
        
        # NOTE: if param file exists, will load params from file and ignore the given params
        if os.path.exists(param_save_path):
            logger.info(f'Loading params from {param_save_path}')
            params = load_params(param_save_path)
            self.dedicated_params = params
            self.universal_params = []
            
        self.prepare_params(workflow.get_all_modules(lambda x: isinstance(x, LangChainLM)), module_mapping)
        self.study = self.init_study()
        
        if os.path.exists(tpe_log_path):
            with open(tpe_log_path, 'r') as f:
                self.tpe_logs = json.load(f)
                for trial_id, meta in self.tpe_logs.items():
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
                workflow=workflow,
                evaluator=evaluator,
                throughput=throughput,
            )
            json.dump(self.tpe_logs, open(tpe_log_path, 'w+'), indent=4)
            params = [param for params in self.params.values() for param in params]
            dump_params(params, param_save_path)
         
        for i, trial in enumerate(self.study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: score= {}, cost= {}".format(f1, f2))
        print("Opt Cost: {}".format(self.opt_cost))
        return self.study.best_trials, self.opt_cost
    
class OuterLoopOptimization:
    class InnerLoopEvaluator(EvaluatorInterface):
        def __init__(
            self,
            inner_loop: InnerLoopBayesianOptimization,
            module_mapping: T_ModuleMapping,
            evaluator: Evaluator,
            n_trials: int,
            quality_constraint: Optional[float] = None,
            throughput: int = 1,
            log_dir: str = 'holm_log',
        ):
            self.evaluator = evaluator
            self.inner_loop = inner_loop
            self.module_mapping = module_mapping
            self.n_trials = n_trials
            self.quality_constraint = quality_constraint
            self.throughput = throughput
            self.log_dir = log_dir
        
        def __call__(
            self,
            workflow: Workflow,
        ) -> EvaluationResult:
            pareto_frontier, opt_cost = self.inner_loop.optimize(
                workflow=workflow,
                module_mapping=self.module_mapping,
                evaluator=self.evaluator,
                n_trials=self.n_trials,
                throughput=self.throughput,
                log_dir=self.log_dir,
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
                states=None,
                demos=None,
                meta={'inner_opt_cost': opt_cost}
            )
            return result
    
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
    
    def prepare_params(self, lm_modules: list[LangChainLM]):
        self.params: dict[str, list[ParamBase]] = defaultdict(list)
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

        self.study: optuna.Study = None
        self._study_lock = threading.Lock()
        self.tpe_logs = {}
        self.best_score_cost = []
    
    def propose(
        self,
        workflow: Workflow,
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, Workflow, T_ModuleMapping]]:
        next_to_run = []
        for i in range(n_sample):
            mapping: T_ModuleMapping = {}
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)
            program = copy.deepcopy(workflow)
            module_dict = {
                lm.name: lm 
                for lm in program.get_all_modules(lambda x: isinstance(x, LangChainLM))
            }
            self.tpe_logs[trial.number] = {'params': {}}
            for lm_name, params in self.params.items():
                for param in params:
                    selected = trial.params[param.hash]
                    _, new_mapping = param.apply_option(selected, module_dict[lm_name])
                    self.tpe_logs[trial.number]['params'][param.hash] = selected
                    mapping = merge_module_mapping(mapping, new_mapping)
                    
            logger.info(f"- OuerLoop - next_to_run - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}")
            program.compile()
            mapping = flatten_mapping(mapping)
            next_to_run.append((trial, program, mapping))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        workflow: Workflow,
        evaluator: Evaluator,
    ):
        eval_result: EvaluationResult = evaluator(workflow)
        for p_score, p_price in zip(eval_result.scores, eval_result.prices):
            self.best_score_cost.append((p_score, p_price))
        inner_cost = eval_result.meta['inner_opt_cost']
        self.opt_cost += inner_cost
        
        score, price = eval_result.reduced_score, eval_result.reduced_price
        logger.info(f"- OuerLoop - Trial {trial.number} result: score: {score}, price: {price}")
        self.tpe_logs[trial.number]['score'] = score
        self.tpe_logs[trial.number]['price'] = price
        self.tpe_logs[trial.number]['total_price'] = inner_cost
         
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
                
    def _opt_loop(
        self, 
        n_outer_trials,
        n_inner_trials,
        workflow,
        evaluator, 
        inner_loop: InnerLoopBayesianOptimization,
        inner_throughput: int,
        log_dir: str,
    ):
        for i in range(n_outer_trials):
            next_trial, program, mapping = self.propose(workflow, 1)[0]
            # use inner loop optimization as evaluator
            feedback_consumer = OuterLoopOptimization.InnerLoopEvaluator(
                inner_loop=copy.deepcopy(inner_loop),
                module_mapping=mapping,
                evaluator=evaluator,
                n_trials=n_inner_trials,
                quality_constraint=self.quality_constraint,
                throughput=inner_throughput,
                log_dir=os.path.join(log_dir, 'inner_loops', uuid.uuid4().hex),
            )
            self._eval_and_update(next_trial, program, feedback_consumer)
    
    def _optimize(
        self, 
        n_outer_trials: int,
        n_inner_trials: int,
        workflow: Workflow, 
        evaluator: Evaluator,
        log_dir: str,
        inner_loop: InnerLoopBayesianOptimization,
        throughput: int, # number of trials to run in parallel
        inner_throughput: int,
    ):
        if throughput == 1:
            self._opt_loop(n_outer_trials, n_inner_trials, workflow, evaluator, inner_loop, inner_throughput, log_dir)
        else:
            futures: set[Future] = set()
            with ThreadPoolExecutor(max_workers=throughput) as executor:
                for n_submitted_trials in range(n_outer_trials):
                    if len(futures) >= throughput:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in completed:
                            f.result()
                    futures.add(executor.submit(self._opt_loop, 1, n_inner_trials, workflow, evaluator, inner_loop, inner_throughput, log_dir))
                    
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
        workflow: Workflow,
        inner_loop: InnerLoopBayesianOptimization,
        evaluator: Evaluator,
        n_trials: int,
        resource_ratio: float = 1 / 10, # trials in outer vs inner loop
        throughput: int = 1,
        inner_throughput: int = 1,
        log_dir: str = 'holm_log',
    ):
        self.opt_cost = 0
        self.outer_loop_log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        workflow.reset()
        tpe_log_path = os.path.join(self.outer_loop_log_dir, 'outer_tpe_logs.json')
        param_save_path = os.path.join(self.outer_loop_log_dir, 'outer_params.json')
        
        if os.path.exists(param_save_path):
            logger.info(f'Loading outerloop params from {param_save_path}')
            params = load_params(param_save_path)
            self.dedicated_params = params
            self.universal_params = []
        
        self.prepare_params(workflow.get_all_modules(lambda x: isinstance(x, LangChainLM)))
        self.study = self.init_study()
        
        if os.path.exists(tpe_log_path):
            with open(tpe_log_path, 'r') as f:
                self.tpe_logs = json.load(f)
                for trial_id, meta in self.tpe_logs.items():
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
                workflow=workflow,
                evaluator=evaluator,
                log_dir=log_dir,
                inner_loop=inner_loop,
                throughput=throughput,
                inner_throughput=inner_throughput,
            )
            json.dump(self.tpe_logs, open(tpe_log_path, 'w+'), indent=4)
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
    
    
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from smac import HyperparameterOptimizationFacade as HPO
from smac import Scenario
from smac.multi_objective.parego import ParEGO

    
class SMACAllInOneLayer:
    def __init__(
        self,
        params: Union[dict[str, list[ParamBase]], list[ParamBase]],
        opt_direction: Literal['maximize', 'minimize'],
        target_modules: Iterable[str] = None,
        fields_in_interest: list[str] = None,
    ):
        self.raw_params = params
        self.opt_direction = opt_direction
        self.fields_in_interest = fields_in_interest
        self.target_modules = set(target_modules) if target_modules is not None else None
        self.opt_cost = 0
    
    def prepare_params(self, workflow: Workflow) -> ConfigurationSpace:
        lm_modules = workflow.get_all_modules(lambda x: isinstance(x, LangChainLM))
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
        self.cs = ConfigurationSpace(seed=42)
        smac_params = []
        for _, params in self.params.items():
            for param in params:
                # TODO: add dependencies when adding decomposition
                smac_param = Categorical(
                    name=param.hash, 
                    items=list(param.options.keys()), 
                    default=param.default_option
                )
                smac_params.append(smac_param)
        self.cs.add(smac_params)
        
    def get_objective_function(self, evaluator: Evaluator, workflow: Workflow):
        def eval(config: Configuration, seed: int) -> Tuple[float, float]:
            candidate = copy.deepcopy(workflow)
            lm_modules = candidate.get_all_modules(lambda x: isinstance(x, LangChainLM))
            module_dict: dict[str, LangChainLM] = {lm.name: lm for lm in lm_modules}
            
            for lm_name, params in self.params.items():
                for param in params:
                    id = param.hash
                    selected = config[id]
                    new_lm, mapping = param.apply_option(selected, module_dict[lm_name])
                    module_dict[lm_name] = new_lm
            
            states, score, price = evaluator(candidate)
            self.opt_cost += price
            if self.opt_direction == 'maximize':
                return {'1-score': 1-score, 'price': price}
            return {'score': score, 'price': price}
        return eval
    
    def optimize(
        self,
        workflow: Workflow,
        evaluator: Evaluator,
        n_trials: int,
        log_dir: str = 'holm_log',
    ):
        """Use SMAC3 to optimize the given workflow
        """
        self.opt_cost = 0
        opt_loop_log_dir = os.path.join(log_dir, 'opt_loop')
        if not os.path.exists(opt_loop_log_dir):
            os.makedirs(opt_loop_log_dir, exist_ok=True)
        
        workflow.reset()
        self.prepare_params(workflow)
        obj_func = self.get_objective_function(evaluator=evaluator, workflow=workflow)
        
        if self.opt_direction == 'maximize':
            objectives = ['1-score', 'price']
        else:
            objectives = ['score', 'price']
        scenario = Scenario(
            self.cs, 
            deterministic=True, 
            n_trials=n_trials,
            output_directory=opt_loop_log_dir,
            objectives=objectives,
        )
        
        initial_config = self.cs.get_default_configuration()
        initial_design = HPO.get_initial_design(
            scenario=scenario,
            n_configs=0,
            additional_configs=[initial_config],
        )
        hpo = HPO(
            scenario=scenario,
            target_function=obj_func,
            initial_design=initial_design,
        )
        configs = hpo.optimize()
        for config in configs:
            print(config)
