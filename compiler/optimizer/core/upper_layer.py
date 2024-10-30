import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, get_type_hints
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
from compiler.optimizer.core.unified_layer_opt import OptimizationLayer, BottomLevelTrialLog

logger = logging.getLogger(__name__)


class LayerEvaluator(GeneralEvaluatorInterface):
    def __init__(
        self,
        target_layer: OptimizationLayer,
        quality_constraint: float = None,
    ):
        self.target_layer = target_layer
        self.quality_constraint = quality_constraint
    
    def evaluate(
        self, 
        layer_task: TopDownInformation, 
        show_process=False
    ) -> EvaluationResult:
        #NOTE: optimization will change layer meta, make a copy
        target_layer_cpy = copy.deepcopy(self.target_layer)
        eval_cost, pareto_frontier, opt_logs = target_layer_cpy.optimize(layer_task)
        inner_log_ids, scores, prices, exec_times = [], [], [], []
        for trial_log in opt_logs.values():
            inner_log_ids.append(trial_log.id)
            scores.append(trial_log.score)
            prices.append(trial_log.price)
            exec_times.append(trial_log.eval_cost)
        reduced_score = max(scores)
        if self.quality_constraint is not None:
            """
            Consider cheapest price that passes the quality constraint
            if no trial passed the quality constraint, set the price to a agreed number, it will be rejected at TPE side no matter what this number is
            """
            passed_price = [price for i, price in enumerate(prices) 
                            if scores[i] >= self.quality_constraint]
            if not passed_price:
                reduced_price = float(0xdeadbeef)
            else:
                reduced_price = min(passed_price)
        else:
            reduced_price = min(prices)
        result = EvaluationResult(
            ids=inner_log_ids,
            scores=scores,
            prices=prices,
            exec_times=exec_times,
            total_eval_cost=eval_cost,
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            demos={'config_log_path': layer_task.opt_config.opt_log_path},
        )
        return result

class SuccessiveHalving:
    """Policy for successive halving resource allocation
    
    If a layer (A) adopts this policy, it will allocate resources for the next layer (B)
    
    layer A will propose `layer_a.opt_config.throughput` trials in each iteration
    
    **SH routine**:
    1. each remaining trial at layer B will have a fixed step_budget = `layer_b.opt_config.n_trials`
    2. after all trials are evaluated, lower half of the trials will be pruned
    3. repeat from step 1 until no trials left
    """
    def __init__(
        self,
        selected_runs: list[Tuple[LayerEvaluator, TopDownInformation]],
    ):
        self.selected_runs = selected_runs
        self.ready_to_run = [i for i in range(len(selected_runs))]
        self.num_inner_trials = [0] * len(selected_runs)
        
    def run_and_prune(self):
        logger.info(f"SH next with {self.ready_to_run}")
        for i in self.ready_to_run:
            tdi = self.selected_runs[i][1]
            self.num_inner_trials[i] += tdi.opt_config.n_trials
            
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=len(self.ready_to_run)) as executor:
            for i in self.ready_to_run:
                futures.append(executor.submit(
                    self.selected_runs[i][0].evaluate, self.selected_runs[i][1]
                ))
                
            try:
                outer_indicators = []
                for f in futures:
                    eval_result: EvaluationResult = f.result()
                    outer_indicators.append((eval_result.reduced_score, eval_result.reduced_price))
            except Exception as e:
                logger.error(f"Error in SH: {e}")
                raise
        
        # outer_indicators = []
        # for i in self.ready_to_run:
        #     eval_result = self.selected_outer_runs[i](self.inner_step_budget)
        #     outer_indicators.append((eval_result.reduced_score, eval_result.reduced_price))
        
        # sort by score and price, higher score then lower price
        sorted_indicator_indices = sorted(
            range(len(outer_indicators)),
            key=lambda i: (-outer_indicators[i][0], outer_indicators[i][1])
        )
        runs_left_to_run = sorted_indicator_indices[:len(self.ready_to_run) // 2]
        self.ready_to_run = [self.ready_to_run[i] for i in runs_left_to_run]
    
    def execute(self):
        while len(self.ready_to_run) > 0:
            self.run_and_prune()
        # Collect inner loop performance
        outer_run_evals = []
        for layer_eval, tdi in self.selected_runs:
            tdi.opt_config.n_trials = 0
            outer_run_evals.append(layer_eval.evaluate(tdi))
        return outer_run_evals, self.num_inner_trials
    
class UpperLevelTrialLog(TrialLog):
    def __init__(
        self, 
        params, 
        bo_trial_id, 
        id = None,
        score = 0, 
        price = 0, 
        eval_cost = 0,
        next_level_log_dir = None,
        num_next_level_trials = 0,
    ):
        super().__init__(params, bo_trial_id, id, score, price, eval_cost)
        self.next_level_log_dir = next_level_log_dir
        self.num_next_level_trials = num_next_level_trials
    
    def to_dict(self):
        return {
            **super().to_dict(),
            'next_level_log_dir': self.next_level_log_dir,
            'num_next_level_trials': self.num_next_level_trials,
        }
    
class UpperLevelOptimization(OptimizationLayer):
    opt_logs: dict[str, UpperLevelTrialLog]
    evaluator: LayerEvaluator
    trial_log_cls = UpperLevelTrialLog
    
    def __init__(
        self, 
        next_level_opt_config: OptConfig = None,
        use_SH_allocation: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.next_level_opt_config = next_level_opt_config
        self.use_SH_allocation = use_SH_allocation
    
    def prepare_next_level_tdi(self, new_program, new_trace, trial_number):
        next_level_info = super().prepare_next_level_tdi(new_program, new_trace)
        # reset opt_config for next level
        if self.next_level_opt_config:
            next_level_info.opt_config.update(self.next_level_opt_config)
        # incase log_dir is not set
        if self.next_level_opt_config.log_dir is None:
            current_level_log_dir = self.top_down_info.opt_config.log_dir
            next_level_info.opt_config.log_dir = os.path.join(
                current_level_log_dir, 
                f"{self.name}_trial_{trial_number}_{self.evaluator.target_layer.name}"
            )
        next_level_info.trace_back.append(self.generate_trial_id(trial_number))
        # set these path to None to let the next level to populate
        next_level_info.opt_config.opt_log_path = None
        next_level_info.opt_config.param_save_path = None
        return next_level_info

    def _optimize_iteration(self, base_program):
        next_trial, program, new_trace, log_id = self.propose(base_program, 1)[0]
        next_level_info = self.prepare_next_level_tdi(program, new_trace)
        
        self.opt_logs[log_id].next_level_log_dir = next_level_info.opt_config.log_dir

        # run evaluation
        try:
            eval_result = self.evaluator.evaluate(next_level_info)
            self.update(next_trial, eval_result, log_id)
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            raise
    
    def _optimize_SH(self, base_program):
        opt_config = self.top_down_info.opt_config
        
        # propose `throughput` trials
        proposals_at_this_level = self.propose(base_program, opt_config.throughput)
        selected_runs = []
        for new_trial, new_program, new_trace, new_log_id in proposals_at_this_level:
            next_level_info = self.prepare_next_level_tdi(new_program, new_trace, new_trial.number)
            selected_runs.append((self.evaluator, next_level_info))
            self.opt_logs[new_log_id].next_level_log_dir = next_level_info.opt_config.log_dir
        sh = SuccessiveHalving(selected_runs)
        eval_results, num_inner_trials = sh.execute()
        for i, (trial, program, new_trace, log_id) in enumerate(proposals_at_this_level):
            self.update(trial, eval_results[i], log_id)
            self.opt_logs[log_id].num_next_level_trials = num_inner_trials[i]
            
    def _optimize(self, base_program):
        if not self.use_SH_allocation:
            return super()._optimize(base_program)
        
        # use SH allocation
        opt_config = self.top_down_info.opt_config
        n_iters = opt_config.n_trials // opt_config.throughput
        for i in range(n_iters):
            self._optimize_SH(base_program)
            if self.save_ckpt_interval > 0 and i % self.save_ckpt_interval == 0:
                self.save_ckpt(opt_config.opt_log_path, opt_config.param_save_path)
    
    def get_all_candidates(self) -> list[TrialLog, str]:
        candidates = []
        for log_id, trial_log in self.opt_logs.items():
            bot_opt_log_path = os.path.join(trial_log.next_level_log_dir, 'opt_logs.json') 
            with open(bot_opt_log_path, 'r') as f:
                opt_trace = json.load(f)
                
            for trial_log_id, trial_meta in opt_trace.items():
                trial_log = BottomLevelTrialLog.from_dict(trial_meta)
                candidates.append((trial_log, bot_opt_log_path))
        return candidates