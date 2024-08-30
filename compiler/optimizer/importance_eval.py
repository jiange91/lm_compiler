import os
import json
from typing import Union, Optional, Any, Tuple, Iterable, Callable
import copy
import logging
import numpy as np
from collections import defaultdict

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor

logger = logging.getLogger(__name__)

class LMImportanceEvaluator:
    """
    For each LM Module, we want to know how important it is to the final output.
    Try all models at each LM while keep the other LMs fixed.
    Estimate the max diff in final output metric.
    
    Args:
        workflow: Workflow
        
        models: list[str]
            list of models with different capabilities
        
        base_model: str
            the base model to be used for other steps other than the current LM
            
        final_output_metric: Callable[[Any, Any], Any]
            always only return one numerical value
            
        trainset_input: Iterable[StatePool]
        
        trainset_label: Iterable[Any]
    """
    def __init__(
        self,
        workflow: Workflow,
        models: list[str],
        base_model: str,
        final_output_metric: Callable[[Any, StatePool], Any],
        trainset_input: Iterable[StatePool],
        trainset_label: Iterable[Any],
    ):
        self.workflow = workflow
        self.models = models
        self.base_model = base_model
        self.final_output_metric = final_output_metric
        self.trainset_input = trainset_input
        self.trainset_label = trainset_label
        
        self.lm_modules: list[LLMPredictor] = workflow.get_all_modules(lambda x: isinstance(x, LLMPredictor))
    
    def hash_run(self, rid):
        lms = [str(rid)]
        for lm in self.lm_modules:
            lms.append(lm.lm_config['model'])
        return "#".join(lms)
        
    def eval(
        self,
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
            lm_importance_by_req: dict[str, list[float]] = defaultdict(list)
            for lm in self.lm_modules:
                lm.lm_config['model'] = self.base_model
            memo = {}
            for lm in self.lm_modules:
                logger.info(f"Eval module: {lm.name}")
                for rid, (state, label) in enumerate(zip(self.trainset_input, self.trainset_label)):
                    min_metric, max_metric = float('inf'), float('-inf')
                    for model in self.models:
                        state_cpy = copy.deepcopy(state)
                        self.workflow.reset()
                        lm.lm_config['model'] = model
                        
                        if (hash := self.hash_run(rid)) in memo:
                            metric = memo[hash]
                        else:
                            self.workflow.pregel_run(state_cpy)
                            metric = self.final_output_metric(label, state_cpy)
                            memo[hash] = metric
                        min_metric = min(min_metric, metric)
                        max_metric = max(max_metric, metric)
                        lm.lm_config['model'] = self.base_model
                    lm_importance_by_req[lm.name].append(max_metric - min_metric)
            # estiamte importance
            lm_importance: dict[str, float] = {}
            for lm, importances in lm_importance_by_req.items():
                lm_importance[lm] = np.mean(importances)
            json.dump(lm_importance, open(os.path.join(log_dir, 'lm_importance.json'), 'w+'))
        # top quantile is important LMs
        sorted_lm_importance = sorted(lm_importance.items(), key=lambda x: x[1], reverse=True)
        important_lms = [lm for lm, _ in sorted_lm_importance[:int(len(sorted_lm_importance) * quantile)]]
        return important_lms
        