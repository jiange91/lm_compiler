import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable
import copy
import logging
import optunahub
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.utils import get_bill

def default_reduer(xs):
    return sum(xs) / len(xs)

class Evaluator:
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
    ):
        prices = []
        states = []
        scores = []
        for input_state, label in self.eval_set:
            state_cpy = copy.deepcopy(input_state)
            workflow.reset()
            workflow.pregel_run(state_cpy)
            
            workflow.update_token_usage_summary()
            price = get_bill(workflow.token_usage_buffer)[0]
            prices.append(price)
            states.append(state_cpy)
            scores.append(self.metric(label, state_cpy))
        return states, scores, prices
    
    def _multi_thread_run(
        self,
        workflow: Workflow,
        num_thread: int,
    ):
        def routine(input_state, label, program: Workflow):
            state_cpy = copy.deepcopy(input_state)
            program.reset()
            program.pregel_run(state_cpy)
            program.update_token_usage_summary()
            price = get_bill(program.token_usage_buffer)[0]
            return state_cpy, self.metric(label, state_cpy), price
        
        with ThreadPoolExecutor(num_thread) as executor:
            futures = []
            for input_state, label in self.eval_set:
                futures.append(executor.submit(routine, input_state, label, copy.deepcopy(workflow)))
            states, scores, prices = [], [], []
            for future in futures:
                state, score, price = future.result()
                states.append(state)
                scores.append(score)
                prices.append(price)
        return states, scores, prices
    
    def __call__(
        self,
        workflow: Workflow,
    ):
        if self.num_thread == 1:
            states, scores, prices = self._single_thread_run(workflow)
        else:
            states, scores, prices = self._multi_thread_run(workflow, self.num_thread)
        return states, self.score_reducer(scores), self.price_reducer(prices)