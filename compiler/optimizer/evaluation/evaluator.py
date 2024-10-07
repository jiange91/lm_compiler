import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
import copy
import logging
from dataclasses import dataclass
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from abc import ABC, abstractmethod
import multiprocess as mp


import logging

from compiler.IR.base import ComposibleModuleInterface
from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration, TokenUsageSummary, TokenUsage
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.optimizer.params.common import ParamBase
from compiler.optimizer.plugin import OptimizerSchema
from compiler.utils import get_bill
from compiler.optimizer.core.flow import TopDownInformation, ModuleTransformTrace

logger = logging.getLogger(__name__)

def default_reduer(xs):
    return sum(xs) / len(xs)

class EvaluatorConfig:
    """Control what to log
    
    TODO: implement this
    """
    ...
    
# {module_name: demo}
TDemoInTrial = dict[str, Demonstration] 

class EvaluationResult:
    def __init__(
        self,
        scores: Sequence[float],
        prices: Sequence[float],
        reduced_score: Optional[float] = None,
        reduced_price: Optional[float] = None,
        demos: Optional[Sequence[TDemoInTrial]] = None,
        
        meta: Optional[dict] = None,
    ) -> None:
        self.scores = scores
        self.prices = prices
        self.reduced_score = reduced_score
        self.reduced_price = reduced_price
        self.demos = demos
        self.meta = meta
    
    def __str__(self) -> str:
        return f"EvalResult: score={self.reduced_score}, price={self.reduced_price}, {len(self.scores)} samples"

class EvaluatorInterface(ABC):
    
    @abstractmethod
    def __call__(
        self,
        workflow: Workflow,
    ) -> EvaluationResult:
        """Evaluate the workflow with the given metric and eval_set
        
        Should not change the state of given workflow
        """
        ...
    

class Evaluator(EvaluatorInterface):
    def __init__(
        self,
        metric: MetricBase,
        eval_set: Iterable[Tuple[StatePool, Any]], 
        num_thread: int = 1,
        score_reducer: Callable = None,
        price_reducer: Callable = None,
    ) -> None:
        self.metric = metric
        self.eval_set = eval_set
        self.score_reducer = score_reducer if score_reducer is not None else default_reduer
        self.price_reducer = price_reducer if price_reducer is not None else default_reduer
        self.num_thread = num_thread
    
    def _single_thread_run(
        self,
        workflow: Workflow,
    ) -> EvaluationResult:
        program = copy.deepcopy(workflow)
        prices = []
        states = []
        scores = []
        demos = []
        for input_state, label in self.eval_set:
            state_cpy = copy.deepcopy(input_state)
            program.reset()
            program.pregel_run(state_cpy)
            
            program.update_token_usage_summary()
            price = get_bill(program.token_usage_buffer)[0]
            prices.append(price)
            states.append(state_cpy)
            scores.append(self.metric(label, state_cpy))
            
            demo = {}
            for lm in program.get_all_modules(lambda x: isinstance(x, LLMPredictor)):
                demo[lm.name] = lm.get_step_as_example()
            demos.append(demo)
        return EvaluationResult(
            scores=scores, 
            prices=prices, 
            demos=demos)
    
    def _multi_thread_run(
        self,
        workflow: Workflow,
        num_thread: int,
    ) -> EvaluationResult:
        def routine(input_state, label, workflow: Workflow):
            state_cpy = copy.deepcopy(input_state)
            program = copy.deepcopy(workflow)
            program.reset()
            program.pregel_run(state_cpy)
            program.update_token_usage_summary()
            price = get_bill(program.token_usage_buffer)[0]
            
            demo = {}
            for lm in program.get_all_modules(lambda x: isinstance(x, LLMPredictor)):
                demo[lm.name] = lm.get_step_as_example()
            return state_cpy, self.metric(label, state_cpy), price, demo
        
        with ThreadPoolExecutor(num_thread) as executor:
            futures = []
            for input_state, label in self.eval_set:
                futures.append(executor.submit(routine, input_state, label, workflow))
            states, scores, prices, demos = [], [], [], []
            for future in futures:
                state, score, price, demo = future.result()
                states.append(state)
                scores.append(score)
                prices.append(price)
                demos.append(demo)
        return EvaluationResult(
            scores=scores, 
            prices=prices, 
            demos=demos)
    

    def __call__(
        self,
        workflow: Workflow,
    ) -> EvaluationResult:
        """Evaluate the workflow with the given metric and eval_set
        
        Will not change the state of given workflow
        """
        if self.num_thread == 1:
            eval = self._single_thread_run(workflow)
        else:
            eval = self._multi_thread_run(workflow, self.num_thread)
        eval.reduced_score = self.score_reducer(eval.scores)
        eval.reduced_price = self.price_reducer(eval.prices)
        return eval

@dataclass
class EvalTask:
    """Define a task to evaluate the score of a workflow
    """
    # execution env
    script_path: str
    args: list[str] # cmd args to the script
    other_python_paths: list[str]
    
    # transformation meta
    all_params: dict[str, ParamBase] # {param_hash: param}
    module_name_paths: dict[str, list[str]]
    aggregated_proposals: dict[str, dict[str, list[tuple[str, str]]]] # {layer_name: {module_name: [(param, option)]}}
    
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        state.pop('all_params')
        state['all_params_ser'] = {}
        param_hashes = self.all_params.keys()
        for hash in param_hashes:
            state['all_params_ser'][hash] = self.all_params[hash].to_dict()
        return state

    def __setstate__(self, state: dict) -> None:
        param_pool = state.pop('all_params_ser')
        self.__dict__.update(state)
        self.all_params = {}
        # restore params
        for hash, param_dict in param_pool.items():
            t = ParamBase.registry[param_dict['type']]
            self.all_params[hash] = t.from_dict(param_dict)

    def add_PYTHON_PATH(self):
        dir = os.path.dirname(self.script_path)
        if dir not in sys.path:
            sys.path.append(dir)
        if self.other_python_paths is not None:
            for path in self.other_python_paths:
                if path not in sys.path:
                    sys.path.append(path)
                
    def replay_module_transformations(self, ms: list[Module]) -> dict[str, Module]:
        module_pool = {m.name: m for m in ms}
        module_ttrace = ModuleTransformTrace({m.name: type(m) for m in ms})
        
        for layer_name, proposal in self.aggregated_proposals.items():
            for module_name, l_param_option in proposal.items():
                module = module_pool[module_name]
                for param_name, option_name in l_param_option:
                    param_hash = ParamBase.chash(module_name, param_name)
                    param = self.all_params[param_hash]
                    
                    new_module, new_mapping = param.apply_option(option_name, module)
                    
                    for old_name, new_name in new_mapping.items():
                        module_ttrace.add_mapping(old_name, [new_name])
                    module_pool[new_module.name] = new_module
                    module = new_module
        
        # check if modules are transformed correctly
        # check equivalence of current name path and registered name path
        assert module_ttrace.eq_transform_path(self.module_name_paths), "Module transformation not consistent"
        
        module_ttrace.mflatten()
        new_modules_dict = {
            ori_name: module_pool[new_name] 
            for ori_name, new_name in module_ttrace.flattened_name_paths.items()
        }
        return new_modules_dict
                    
    def evaluate_program(self, input, label):
        self.add_PYTHON_PATH()
        sys.argv = [self.script_path] + self.args
        schema = OptimizerSchema.capture(self.script_path)
        logger.debug(f'opt_target_modules = {schema.opt_target_modules}')
        assert schema.opt_target_modules, "No module to optimize"
        module_pool = {m.name: m for m in schema.opt_target_modules}
        
        # replace module invoke with new module
        # this does not replace the model but only the invoke function
        if self.aggregated_proposals is not None:
            new_modules_dict = self.replay_module_transformations(schema.opt_target_modules)
            for m in schema.opt_target_modules:
                new_module = new_modules_dict[m.name]
                if isinstance(new_module, Workflow):
                    new_module.compile()
                logger.debug(f'replace {m} with {new_module}')
                m.invoke = new_module.invoke
                m.reset()
                module_pool[m.name] = new_module
            
        result = schema.program(input)
        score = schema.score_fn(label, result)
        
        # get price and demo of running the program
        usages = []
        lm_2_demo = {}
        for lm in Module.all_of_type(module_pool.values(), LLMPredictor):
            usages.extend(lm.get_token_usage())
            lm_2_demo[lm.name] = lm.get_step_as_example()
        
        summary = TokenUsageSummary.summarize(usages)
        price = summary.total_price
        return result, score, price, lm_2_demo

    @classmethod
    def from_top_down_info(cls, tdi: TopDownInformation):
        return cls(
            script_path=tdi.script_path,
            args=tdi.script_args,
            other_python_paths=tdi.other_python_paths,
            all_params=tdi.all_params,
            module_name_paths=tdi.module_ttrace.module_name_paths,
            aggregated_proposals=tdi.module_ttrace.aggregated_proposals,
        )
    
class GeneralEvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: EvalTask,
    ) -> EvaluationResult:
        ...

class EvaluatorPlugin(GeneralEvaluatorInterface):
    def __init__(
        self,
        eval_set: Iterable[Tuple[any,any]], # list of input data and labels
        n_parallel: int = 1,
        score_reducer: Callable = None,
        price_reducer: Callable = None,
    ):
        self.eval_set = eval_set
        self.n_parallel = n_parallel
        self.score_reducer = score_reducer if score_reducer is not None else default_reduer
        self.price_reducer = price_reducer if price_reducer is not None else default_reduer
        
    def evaluate(
        self,
        task: EvalTask,
    ):
        task.add_PYTHON_PATH()
        logger.debug(f'sys_path = {sys.path}')
        
        with mp.Pool(processes=self.n_parallel) as pool:
            tasks = []
            for input, label in self.eval_set:
                tasks.append(
                    pool.apply_async(task.evaluate_program, args=(input, label))
                )
            results = [task.get() for task in tasks]
            
        prices = []
        scores = []
        demos = []
        for result, score, price, demo in results:
            prices.append(price)
            scores.append(score)
            demos.append(demo)
        reduced_score = self.score_reducer(scores)
        reduced_price = self.price_reducer(prices)
        return EvaluationResult(
            scores=scores,
            prices=prices,
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            demos=demos,
        )
    