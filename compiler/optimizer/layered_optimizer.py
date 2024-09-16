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
import threading

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill
from compiler.optimizer.tracer import batch_run_and_eval, OfflineBatchTracer
from compiler.optimizer.params.common import ParamBase, OptionBase, DynamicParamBase, EvolveType
from compiler.optimizer.params.utils import dump_params
from compiler.optimizer.params.model_selection import LMSelection
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.evaluator import Evaluator, EvaluationResult
from optuna.samplers import TPESampler


logger = logging.getLogger(__name__)

class OptRoutineBase(ABC):
    
    @abstractmethod
    def propose(self, n_sample) -> Tuple[list[optuna.trial.Trial], list[Workflow]]:
        """Based on existing observations, propose new configs
        """
        pass
    
    @abstractmethod
    def update(self, feedback):
        """Consume feedbacks from rear layers and update search strategy
        """
        pass

class LLMOptRoutine(OptRoutineBase):
    def __init__(
        self,
        workflow: Workflow,
        opt_direction: Literal['maximize', 'minimize'],
        params: dict[str, list[ParamBase]],
        quality_constraint: Callable[[float], bool],
    ):
        self.workflow = workflow
        self.opt_direction = opt_direction
        self.study = optuna.create_study(
            directions=[self.opt_direction, 'minimize'], # minimize price
            sampler=TPESampler(multivariate=True)
        )
        self.params = params
        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(list(param.options.keys()))
            for _, params in self.params.items() for param in params
        }
        
        self.quality_constraint = quality_constraint
        self.opt_logs = {}
    
    def propose(self, n_sample) -> Tuple[list[optuna.trial.Trial], list[Workflow]]:
        workflows = []
        trials = []
        for i in range(n_sample):
            trial = self.study.ask(self.param_categorical_dist)
            program = copy.deepcopy(self.workflow)
            module_dict = {lm.name: lm 
                           for lm in program.get_all_modules(lambda x: isinstance(x, LangChainLM))}
            self.opt_logs[trial.number] = {'params': {}}
            for lm_name, params in self.params.items():
                for param in params:
                    selected = trial.params[param.hash]
                    param.apply_option(selected, module_dict[lm_name])
                    self.opt_logs[trial.number]['params'][param.hash] = selected
            workflows.append(program)
            trials.append(trial)
        return trials, workflows

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
    """One layer of optimization
    
    Building block of hierarchical optimization
    """
    def __init__(
        self,
        name: str,
        params: Union[dict[str, list[ParamBase]], list[ParamBase]],
        opt_direction: Literal['maximize', 'minimize'],
        target_modules: Iterable[str] = None,
        fields_in_interest: list[str] = None,
    ):
        """Create a LOPT instance with configs
        
        Args:
            params: a dict mapping param name to a list of 
        """
        self.name = name
        self.raw_params = params
        self.opt_direction = opt_direction
        self.fields_in_interest = fields_in_interest
        self.target_modules = set(target_modules) if target_modules is not None else None
        
    @abstractmethod
    def prepare_params(self, workflow) -> OptRoutineBase:
        pass


class GeneralLLMLayer(LayerBase):
    """General Layer for LLM cost-effective optimization
    """
    def __init__(
        self, 
        name: str,
        params: Union[dict[str, list[ParamBase]], list[ParamBase]],
        quality_constraint: Callable[[float], bool],
        opt_direction: Literal['maximize', 'minimize'],
        target_modules: Iterable[str] = None,
        fields_in_interest: list[str] = None,
    ):
        super().__init__(name, params, opt_direction, target_modules, fields_in_interest)
        self.opt_logs = {}
        self.quality_constraint = quality_constraint
    
    def prepare_params(self, workflow: Workflow) -> OptRoutineBase:
        lm_modules = workflow.get_all_modules(lambda x: isinstance(x, LangChainLM))
        _params: dict[str, list[ParamBase]] = {}
        if isinstance(self.raw_params, list):
            for lm in lm_modules:
                lm_name = lm.name
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                params_cpy = copy.deepcopy(self.raw_params)
                for param in params_cpy:
                    param.module_name = lm_name
                _params[lm_name] = params_cpy
        else:
            for lm_name, params in self.raw_params.items():
                if self.target_modules and lm_name not in self.target_modules:
                    continue
                for param in params:
                    assert param.module_name == lm_name, f"param {param.name} has module_name {param.module_name} not matching {lm_name}"
                _params[lm_name] = self.raw_params[lm_name]
        
        opt_routine = LLMOptRoutine(
            workflow=workflow,
            opt_direction=self.opt_direction,
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
        self.study = None
        self._study_lock = threading.Lock()
    
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
        self.tpe_logs = {}
    
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
            
            logger.info(f"innerloop - start Trial: {trial.number}")
            self.tpe_logs[trial.number] = {}
            self.tpe_logs[trial.number]['params'] = {}
            
            for lm_name, params in self.params.items():
                for param in params:
                    id = param.hash
                    selected = trial.suggest_categorical(id, self.param_categorical_dist[id])
                    new_lm = param.apply_option(selected, module_dict[lm_name])
                    module_dict[lm_name] = new_lm
                    self.tpe_logs[trial.number]['params'][id] = selected
            logger.info(f"innerloop - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}") 
            
            states, score, price = evaluator(candidate)
            self.tpe_logs[trial.number]['score'] = score
            self.tpe_logs[trial.number]['price'] = price
            self.opt_cost += price
            logger.info(f"- Trial {trial.number} result: score: {score}, price: {price}")
            
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
                    
            logger.info(f"next_to_run - Trial {trial.number} params: {self.tpe_logs[trial.number]['params']}")
            next_to_run.append((trial, program))
        return next_to_run

    def _eval_and_update(
        self,
        trial: optuna.trial.Trial,
        workflow: Workflow,
        evaluator: Evaluator,
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
        evaluator: Evaluator,
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
        evaluator: Evaluator,
        n_trials: int,
        throughput: int = 1,
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
            dump_params(params, os.path.join(self.inner_loop_log_dir, 'params.json'))
            
        for i, trial in enumerate(self.study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: f1= {}, f2= {}".format(f1, f2))
        print("Opt Cost: {}".format(self.opt_cost))
        return self.study.best_trials
    
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
                    new_lm = param.apply_option(selected, module_dict[lm_name])
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
