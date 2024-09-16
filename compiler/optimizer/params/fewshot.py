from typing import Literal, Optional, Tuple
import uuid
import dataclasses
import heapq
import os
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

from compiler.IR.base import Module
from compiler.IR.program import Workflow
from compiler.IR.llm import LLMPredictor, Demonstration
from compiler.optimizer.params.common import EvolveType, ParamBase, ParamLevel, OptionBase, DynamicParamBase, IdentityOption
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, inspect_runnable
from compiler.optimizer.evaluation.evaluator import EvaluationResult, Evaluator
from compiler.optimizer.params.utils import dump_params, load_params

    
class LMFewShot(DynamicParamBase):
    level = ParamLevel.NODE
    
    def __init__(
        self, 
        name: str,
        module_name: str,
        max_num: int = 5,
        eval_result: EvaluationResult = None,
    ):
        # NOTE: identity option is added to escape from bad demos
        super().__init__(name, [IdentityOption()], 0, module_name)
        self.demo_pool: dict[str, Demonstration] = {}
        self.demo_pq = []
        self.max_num = max_num
        self.current_best_score_sum = float('-inf')
        if eval_result is not None:
            t = self.evole(eval_result)
            # assert t != EvolveType.ID, 'Should evolve'
            if t == EvolveType.ID:
                Warning(f'Given evaluation result does not contain good demos for {module_name}')
    
    @classmethod
    def bootstrap(
        cls,
        workflow: Workflow,
        evaluator: Evaluator,
        max_num: int = 5,
        target_modules: Optional[list[str]] = None,
        log_path: Optional[str] = None,
    ):
        """Collect good demos for LLMs in a workflow
        
        if target_modules is provided, only collect demos for these modules
        """
        # if log_path exists, load from it
        if log_path is not None:
            if os.path.exists(log_path):
                logger.info(f'Loading from {log_path}')
                return load_params(log_path)
            
        eval_result = evaluator(workflow)
        if target_modules is not None:
            lms = workflow.get_all_modules(lambda x: x.name in target_modules)
            for lm in lms:
                if not isinstance(lm, LLMPredictor):
                    raise ValueError(f'{lm.name} is not a LLMPredictor')
        else:
            lms: list[LLMPredictor] = workflow.get_all_modules(lambda x: isinstance(x, LLMPredictor))
        params = []
        for lm in lms:
            params.append(cls('fewshot_demo', lm.name, max_num, eval_result))
        # if log_path provided, save to it
        if log_path is not None:
            logger.info(f'Saving to {log_path}')
            dir = os.path.dirname(log_path)
            if dir:
                os.makedirs(dir, exist_ok=True)
            dump_params(params, log_path)
        return params

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, max_num, current_best_score_sum = data['name'], data['module_name'], data['max_num'], data['current_best_score_sum']
        param = cls(name, module_name, max_num)
        
        demo_pool = {demo['id']: Demonstration(**demo) for demo in data['demo_pool']}
        demo_l = data['demo_pq']
        demo_pq = [(score, demo_id) for score, demo_id in demo_l]
        
        options = data['options']
        options.pop('Identity', None)
        options = {name: DemoOption.from_dict(option, demo_pool) for name, option in options.items()}
        
        param.demo_pool = demo_pool
        param.demo_pq = demo_pq
        param.current_best_score_sum = current_best_score_sum
        param.options = options
        return param
            
    
    def evole(self, eval_result: EvaluationResult) -> EvolveType:
        """Update demo range given current evaluation result
        
        always select top k demos as new option candidate 
        only accept this candidate if sum of their score is higher than current best option
        
        """
        selection = heapq.nlargest(self.max_num, enumerate(eval_result.scores), key=lambda x: x[1])
        for i, score in selection:
            demo = eval_result.demos[i][self.module_name]
            if demo is not None:
                heapq.heappush(self.demo_pq, (score, demo.id))
                self.demo_pool[demo.id] = demo
            
        score_sum = 0
        demos = []
        for score, demo_id in self.demo_pq[-self.max_num:]:
            score_sum += score
            demos.append(self.demo_pool[demo_id])
        if demos and score_sum > self.current_best_score_sum:
            self.current_best_score_sum = score_sum
            option_name = f'{self.module_name}_demos_{str(uuid.uuid4())}'
            self.add_option(DemoOption(option_name, demos))
            return EvolveType.RANGE
        return EvolveType.ID

    def to_dict(self):
        base = super().to_dict()
        base['demo_pool'] = [dataclasses.asdict(demo) for demo in self.demo_pool.values()]
        base['demo_pq'] = self.demo_pq
        base['max_num'] = self.max_num
        base['current_best_score_sum'] = self.current_best_score_sum
        return base

    
class DemoOption(OptionBase):
    def __init__(self, tag: str, demos: list[Demonstration]):
        super().__init__(tag)
        self.demos = demos
    
    def apply(self, lm_module: LLMPredictor):
        lm_module.semantic.set_demos(self.demos)
        lm_module.reset() # to trigger prompt reconstruction
        return lm_module
    
    def to_dict(self):
        base = super().to_dict()
        base['demo_ref'] = [demo.id for demo in self.demos]
        return base

    @classmethod
    def from_dict(cls, data: dict, demo_pool: dict[str, Demonstration]):
        tag = data['name']
        demo_ref = data['demo_ref']
        demos = [demo_pool[demo_id] for demo_id in demo_ref]
        return cls(tag, demos)