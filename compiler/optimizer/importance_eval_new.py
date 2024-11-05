import os
import json
from typing import Union, Optional, Any, Tuple, Iterable, Callable
import copy
import logging
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.cog_hub.common import CogBase, OptionBase
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.evaluation.metric import MetricBase

logger = logging.getLogger(__name__)

class LMImportanceEvaluator:
    """
    For each LM Module, we want to know how important it is to the final output.
    Try all models at each LM while keep the other LMs fixed.
    Estimate the max diff in final output metric.
    
    When testing the importance of a LM, we will keep the other LMs fixed at the first option.
    """
    def __init__(
        self,
        workflow: Workflow,
        options: Union[dict[str, list[OptionBase]], list[OptionBase]],
        target_modules: Iterable[str] = None,
    ):
        self.workflow = workflow
        
        def is_target(module: Module):
            if target_modules is None:
                return isinstance(module, LangChainLM)
            else:
                return module.name in target_modules
        self.lm_module_names: set[str] = set([lm.name for lm in workflow.get_all_modules(is_target)])
        
        if isinstance(options, list):
            self.options = {}
            for lm_name in self.lm_module_names:
                # NOTE: model option is thread safe
                self.options[lm_name] = options
        else:
            self.options = options
        
    def hash_run(self, rid):
        lms = [str(rid)]
        for lm in self.lm_modules:
            lms.append(lm.lm_config['model'])
        return "#".join(lms)

    def prepare_eval_env(self, target_name: str, option_idx: int):
        program = copy.deepcopy(self.workflow)
        name_2_lm = {lm.name: lm 
                     for lm in program.get_all_modules(lambda x: x.name in self.lm_module_names)}
        
        for lm_name in self.lm_module_names:
            apply_idx = 0
            if lm_name == target_name:
                apply_idx = option_idx
            self.options[lm_name][apply_idx].apply(name_2_lm[lm_name])
        return program
        

    def batch_eval(self, evaluator: Evaluator):
        lm_scores: dict[str, list[float]] = defaultdict(list)
        lm_prices: dict[str, list[float]] = defaultdict(list)
        
        def get_importance_of(lm_name: str):
            for idx in range(len(self.options[lm_name])):
                workfow = self.prepare_eval_env(lm_name, idx)
                workfow.reset()
                states, score, price = evaluator(workfow)
                lm_scores[lm_name].append(score)
                lm_prices[lm_name].append(price)

        with ThreadPoolExecutor() as per_lm_executor:
            for lm_name in self.lm_module_names:
                per_lm_executor.submit(get_importance_of, lm_name)
        return lm_scores, lm_prices
        
    def eval(
        self,
        evaluator: Evaluator,
        log_dir: str = 'importance_eval_log',
        quantile: float = 0.5,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, 'lm_importance.json')
        if os.path.exists(log_path):
            logger.info("lm_importance.json already exists, read and skip sampling")
            with open(log_path) as f:
                lm_importance = json.load(f)
        else:
            # sample metrics
            lm_scores, lm_prices = self.batch_eval(evaluator)
            # estiamte importance
            lm_importance_score: dict[str, float] = {}
            lm_importance_price: dict[str, float] = {}
            for lm_name in lm_scores:
                lm_importance_score[lm_name] = max(lm_scores[lm_name]) - min(lm_scores[lm_name])
                lm_importance_price[lm_name] = max(lm_prices[lm_name]) - min(lm_prices[lm_name])
            lm_importance_score = sorted(lm_importance_score.items(), key=lambda x: x[1], reverse=True)
            lm_importance_price = sorted(lm_importance_price.items(), key=lambda x: x[1], reverse=True)
            lm_importance = {'score': lm_importance_score, 'price': lm_importance_price}
            json.dump(lm_importance, 
                      open(os.path.join(log_dir, 'lm_importance.json'), 'w+'),
                      indent=4)
            
        # top quantile important LMs in scores
        sorted_lm_importance = sorted(lm_importance['score'], key=lambda x: x[1], reverse=True)
        important_lms = [lm for lm, _ in sorted_lm_importance[:int(len(sorted_lm_importance) * quantile)]]
        logger.info(f"Important LMs by score: {important_lms}")
        return important_lms
        