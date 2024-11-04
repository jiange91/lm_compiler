import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable
import copy
import logging
import optunahub
import optuna
import numpy as np

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor


logger = logging.getLogger(__name__)

# TODO: optimize the batch interface
def batch_run_and_eval(
    workflow: Workflow, 
    inputs: Iterable[StatePool], 
    labels: Iterable[Any],
    output_metric: Callable[[Any, StatePool], Any],
):
    prices = []
    states = []
    for state in inputs:
        state_cpy = copy.deepcopy(state)
        workflow.reset()
        workflow.pregel_run(state_cpy)
        
        workflow.update_token_usage_summary()
        price = get_bill(workflow.token_usage_buffer)[0]
        prices.append(price)
        
        states.append(state_cpy)
    scores = []
    for gt, state in zip(labels, states):
        scores.append(output_metric(gt, state))
    return states, scores, prices

class OfflineBatchTracer:
    def __init__(
        self,
        workflow: Workflow,
        module_2_config: Union[dict[str, str], str],
        final_output_metric: Callable[[Any, StatePool], Any],
    ) -> None:
        self.workflow = workflow
        self.lm_modules: list[LLMPredictor] = workflow.get_all_modules(lambda x: isinstance(x, LLMPredictor))
        if isinstance(module_2_config, str):
            module_2_config = {m.name: module_2_config for m in self.lm_modules}
        self.module_2_config = module_2_config
        self.final_output_metric = final_output_metric
        
    def run(
        self,
        inputs: Iterable[StatePool],
        labels: Iterable[Any],
        field_in_interest: list[str] = None,
        log_dir: str = 'trace_log',
    ):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'trials.json')
        if os.path.exists(log_path):
            logger.info(f"Loading tracing results from {log_path}")
            with open(log_path, 'r') as f:
                trials = json.load(f)
            return trials
        
        for lm in self.lm_modules:
            lm.lm_config['model'] = self.module_2_config[lm.name]
        states, scores, price = batch_run_and_eval(self.workflow, inputs, labels, self.final_output_metric)
        if field_in_interest is not None:
            trials = [{'fields': state.all_news(field_in_interest), 'score': score, 'price': price} for state, score, price in zip(states, scores, price)]
        else:
            trials = [{'score': score, 'price': price} for state, score, price in zip(states, scores, price)]
        json.dump(trials, open(log_path, 'w+'), indent=4)
        logger.info(f"Dumped tracing results to {log_path}")
        return trials