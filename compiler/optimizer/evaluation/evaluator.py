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
import multiprocessing as mp


import logging

from compiler.IR.base import ComposibleModuleInterface
from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration, TokenUsageSummary, TokenUsage
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.optimizer.plugin import OptimizerSchema
from compiler.utils import get_bill

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
    
    module_pool should include all the modules that are to be used in the workflow
    """
    script_path: str
    args: list[str] # cmd args to the script
    module_map_table: dict[str, str]
    module_pool: dict[str, Module]
    other_python_paths: Optional[list[str]] = None
    
    def add_PYTHON_PATH(self):
        dir = os.path.dirname(self.script_path)
        if dir not in sys.path:
            sys.path.append(dir)
        if self.other_python_paths is not None:
            for path in self.other_python_paths:
                if path not in sys.path:
                    sys.path.append(path)
    
    def evaluate_program(self, input: dict, label: dict):
        self.add_PYTHON_PATH()
        sys.argv = [self.script_path] + self.args
        schema = OptimizerSchema.capture(self.script_path)
        logger.debug(f'opt_target_modules = {schema.opt_target_modules}')
        assert schema.opt_target_modules, 'No optimize target modules found in the script'
        
        # replace module invoke with new module
        for m in schema.opt_target_modules:
            if self.module_pool:
                if m.name in self.module_map_table:
                    new_module = self.module_pool[self.module_map_table[m.name]]
                else:
                    new_module = self.module_pool[m.name]
            else:
                continue
            if isinstance(new_module, Workflow):
                new_module.compile()
            logger.debug(f'replace {m} with {new_module}')
            m.invoke = new_module.invoke
            m.reset()
            
        if self.module_pool is None:
            self.module_pool = {m.name: m for m in schema.opt_target_modules} 
            
        result = schema.program(input)
        score = schema.score_fn(label, result)
        
        # get price and demo of running the program
        usages = []
        lm_2_demo = {}
        for lm in Module.all_of_type(self.module_pool.values(), LLMPredictor):
            usages.extend(lm.get_token_usage())
            lm_2_demo[lm.name] = lm.get_step_as_example()
        
        summary = TokenUsageSummary.summarize(usages)
        price = summary.total_price
        return result, score, price, lm_2_demo
    
# class NoDaemonProcess(mp.Process):
#     @property
#     def daemon(self):
#         return False

#     @daemon.setter
#     def daemon(self, value):
#         pass


# class NoDaemonContext(type(mp.get_context(method="spawn"))):
#     Process = NoDaemonProcess

# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(mp.pool.Pool):
#     def __init__(self, *args, **kwargs):
#         kwargs['context'] = NoDaemonContext()
#         super(MyPool, self).__init__(*args, **kwargs)

class EvaluatorPlugin:
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
    