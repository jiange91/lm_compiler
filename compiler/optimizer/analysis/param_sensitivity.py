import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Iterable, Callable, Type
import copy
import logging
import numpy as np
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.optimizer.params.common import ParamBase, OptionBase
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.optimizer.evaluation.evaluator import EvaluatorPlugin, EvalTask, EvaluationResult

logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """
    For each Module, we want to know how sensitive it is to a specific parameter.
    
    1. Try all options at each module whiling keep the other modules fixed at default option.
    2. Estimate the max diff in final output metric.
    
    if try_options is given, will choose from the try_options instead of `param.options`
    """
    def __init__(
        self,
        target_param_type: Type[ParamBase],
        eval_task: EvalTask,
        evaluator: EvaluatorPlugin,
        n_parallel: int = 1,
        log_dir: str = None,
        try_options: Optional[ParamBase] = None,
        module_type: Type[Module] = None,
    ):
        self.target_param_type = target_param_type
        self.eval_task = eval_task
        self.evalutor = evaluator
        self.n_parallel = n_parallel
        self.log_dir = log_dir
        self.try_options = try_options
        self.module_type = module_type
        if try_options and not isinstance(self.try_options, self.target_param_type):
            raise ValueError(f'try_options should be of type {self.target_param_type}')
        if not self.eval_task.aggregated_proposals:
            if self.module_type is None:
                raise ValueError('module_type should be provided if aggregated_proposals is not given')
            if self.try_options is None:
                raise ValueError('try_options should be provided if aggregated_proposals is not given')
    
    def run(self, quantile: float = 0.5):
        """get top quantile sensitive modules in scores w.r.t. the target param
        """
        logger.info(f'Starting sensitivity analysis for {self.target_param_type.__name__}')
        log_path = os.path.join(self.log_dir, f'{self.target_param_type.__name__}_sensitivity.json')
        if os.path.exists(log_path):
            logger.info(f'loading from existing log {log_path}')
            with open(log_path, 'r') as f:
                overall_sensitive = json.load(f)
                m_score_sensitivity = overall_sensitive['score']
                m_price_sensitivity = overall_sensitive['price']
        else:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
            tasks = self.spawn_tasks() # first one will always be the base
            m_scores: dict[str, list[float]] = defaultdict(list)
            m_prices: dict[str, list[float]] = defaultdict(list)
            
            logger.info(f'Running {len(tasks)} sensitivity tasks')
            futures: list[Future] = []
            with ThreadPoolExecutor(max_workers=min(len(tasks), self.n_parallel)) as executor:
                for tag, task in tasks:
                    future = executor.submit(self.evalutor.evaluate, task)
                    futures.append(future)
                try:
                    base_score, base_price = 0, 0
                    for i, ((tag, _), f) in enumerate(zip(tasks, futures)):
                        eval_result: EvaluationResult = f.result()
                        if i == 0:
                            base_score, base_price = eval_result.reduced_score, eval_result.reduced_price
                        else:
                            m_scores[tag].append(eval_result.reduced_score)
                            m_prices[tag].append(eval_result.reduced_price)
                    for m_name in m_scores:
                        m_scores[m_name].append(base_score)
                        m_prices[m_name].append(base_price)
                except Exception as e:
                    logger.error(f'Error in evaluating {task}')
                    raise
            
            m_score_sensitivity, m_price_sensitivity = {}, {}
            for m_name, scores in m_scores.items():
                m_score_sensitivity[m_name] = max(scores) - min(scores)
            for m_name, prices in m_prices.items():
                m_price_sensitivity[m_name] = max(prices) - min(prices)
            m_score_sensitivity = sorted(m_score_sensitivity.items(), key=lambda x: x[1], reverse=True)
            m_price_sensitivity = sorted(m_price_sensitivity.items(), key=lambda x: x[1], reverse=True)
            overall_sensitive = {'score': m_score_sensitivity, 'price': m_price_sensitivity}
            json.dump(overall_sensitive, open(log_path, 'w+'), indent=4)
        
        # top quantile important modules in scores
        sensitive_ms = [m_name for m_name, score in m_score_sensitivity[:int(len(m_score_sensitivity) * quantile)]]
        logger.info(f"Sensitive modules by score: {sensitive_ms}")
        return sensitive_ms
    
    def spawn_tasks(self) -> list[tuple[str, EvalTask]]:
        """generate tasks for sensitivity analysis
        """
        new_proposals = []
        if self.eval_task.all_params:
            param_pool = self.eval_task.all_params.copy()
        else:
            param_pool = {}
        
        # create base proposal with all target param set to default option
        if not self.eval_task.aggregated_proposals:
            module_2_params = {}
            base_aggregated_proposals = {f'{self.target_param_type.__name__}_sensitivity_layer': module_2_params}
            schema = self.eval_task.get_program_schema()
            target_modules = Module.all_of_type(schema.opt_target_modules, self.module_type)
            for m in target_modules:
                module_2_params[m.name] = [(self.try_options.name, self.try_options.get_default_option().name)]
                sen_param = copy.deepcopy(self.try_options)
                sen_param.module_name = m.name
                param_pool[sen_param.hash] = sen_param
        else:
            base_aggregated_proposals = copy.deepcopy(self.eval_task.aggregated_proposals)
            for layer_name, m_2_proposals in base_aggregated_proposals.items():
                for module_name, proposals in m_2_proposals.items():
                    for i, (param_name, option_name) in enumerate(proposals):
                        param_hash = ParamBase.chash(module_name, param_name)
                        param = param_pool[param_hash]
                        if isinstance(param, self.target_param_type):
                            # identify the target
                            if self.try_options:
                                try_from = copy.deepcopy(self.try_options)
                                try_from.module_name = module_name
                                param_pool[try_from.hash] = try_from
                            else:
                                try_from = param
                            proposals[i] = (try_from.name, try_from.get_default_option().name)
        new_proposals.append(
            (
                f'{self.target_param_type.__name__}_sensitivity_base', 
                base_aggregated_proposals
            )
        )
        
        # create sensitivity tasks
        for layer_name, m_2_proposals in base_aggregated_proposals.items():
            for module_name, proposals in m_2_proposals.items():
                for i, (param_name, option_name) in enumerate(proposals):
                    param_hash = ParamBase.chash(module_name, param_name)
                    param = param_pool[param_hash]
                    if isinstance(param, self.target_param_type):
                        # identify the target
                        default_option = option_name
                        for other_option in param.options:
                            if other_option != default_option:
                                proposals[i] = (param_name, other_option)
                                new_proposals.append((module_name, copy.deepcopy(base_aggregated_proposals)))
                        # restore the default option
                        proposals[i] = (param_name, default_option)
        
        new_tasks: list[tuple[str, EvalTask]] = []
        for tag, proposal in new_proposals:
            new_task = copy.deepcopy(self.eval_task)
            new_task.all_params = param_pool
            new_task.aggregated_proposals = proposal
            new_tasks.append((tag, new_task))
        return new_tasks