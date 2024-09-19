import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
import copy
import logging
import optunahub
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from abc import ABC, abstractmethod

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.utils import get_bill

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
        states: Optional[Sequence[StatePool]] = None,
        demos: Optional[Sequence[TDemoInTrial]] = None,
        
        meta: Optional[dict] = None,
    ) -> None:
        self.scores = scores
        self.prices = prices
        self.reduced_score = reduced_score
        self.reduced_price = reduced_price
        self.states = states
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
            states=states, 
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
            states=states, 
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
