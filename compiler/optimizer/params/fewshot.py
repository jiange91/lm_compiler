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
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.params.utils import dump_params, load_params

    
class LMFewShot(DynamicParamBase):
    level = ParamLevel.NODE
    
    def __init__(
        self, 
        name: str,
        max_num: int = 5,
        module_name: str = None,
        eval_result: EvaluationResult = None,
        inherit: bool = False,
        allow_duplicate: bool = False,
    ):
        # NOTE: identity option is added to escape from bad demos
        super().__init__(name, [IdentityOption()], 0, module_name, inherit=inherit, inherit_options=False)
        # cached good demos in all options
        # demo_id -> Demonstration
        self.demo_cache: dict[str, Demonstration] = {}
        # task_id -> score
        self.best_score_by_task: dict[str, float] = {}
        # priority queue for demos (score, task_id, demo_id)
        self.demo_pq: list[tuple[float, str, str]] = []
        # task_id in priority queue
        self.task_id_set = set()
        
        self.max_num = max_num
        self.current_best_score_sum = float('-inf')
        self.allow_duplicate = allow_duplicate
        if eval_result is not None:
            t = self.evole(eval_result)
            # Some agent might not have a single demo so this assert is not valid
            # assert t != EvolveType.ID, 'Should evolve'
            if t == EvolveType.ID:
                Warning(f'Given evaluation result does not contain good demos for {module_name}')
    

    def evole(self, eval_result: EvaluationResult) -> EvolveType:
        """Update demo range given current evaluation result
        
        always select top k demos as new option candidate 
        only accept this candidate if sum of their score is higher than current best option
        
        """
        # update demo pool
        updated = set()
        demo_pool: dict[int, Demonstration] = {}
        for task_id, demo, score in zip(eval_result.ids, eval_result.demos, eval_result.scores):
            demo_pool[task_id] = demo[self.module_name]
            # NOTE: use < to prevent too frequent update of same task demo
            # also this check requires demo to surpass itself even for allow_duplicate
            if task_id not in self.best_score_by_task or self.best_score_by_task[task_id] < score: 
                updated.add(task_id)
                self.best_score_by_task[task_id] = score
        
        # update priority queue
        new_option = False
        for task_id in updated:
            score, demo = self.best_score_by_task[task_id], demo_pool[task_id]
            if len(self.demo_pq) < self.max_num:
                heapq.heappush(self.demo_pq, (score, task_id, demo.id))
                self.task_id_set.add(task_id)
                new_option = True
                continue
            
            # if allow_duplicate or not added yet, directly replace the lowest score demo
            if self.allow_duplicate:
                # more strict condition for allow_duplicate
                if score > self.demo_pq[0][0]:
                    self.task_id_set.remove(self.demo_pq[0][1])
                    heapq.heapreplace(self.demo_pq, (score, task_id, demo.id))
                    self.task_id_set.add(task_id)
                    new_option = True
            elif task_id not in self.task_id_set:
                # prepare trying similar quality demos if different task_id
                if score >= self.demo_pq[0][0]:
                    self.task_id_set.remove(self.demo_pq[0][1])
                    heapq.heapreplace(self.demo_pq, (score, task_id, demo.id))
                    self.task_id_set.add(task_id)
                    new_option = True
            else:
                # same task_id already in the queue, need to replace the same task demo
                new_pq = [e for e in self.demo_pq if e[1] != task_id]
                heapq.heapify(new_pq)
                heapq.heappush(new_pq, (score, task_id, demo.id))
                self.demo_pq = new_pq
                new_option = True
            
        if new_option:
            score_sum = 0
            demos = []
            for score, task_id, demo_id in self.demo_pq:
                score_sum += score
                if demo_id not in self.demo_cache:
                    self.demo_cache[demo_id] = demo_pool[task_id]
                demos.append(self.demo_cache[demo_id])
            self.current_best_score_sum = score_sum
            option_name = f'{self.module_name}_demos_{str(uuid.uuid4())}'
            self.add_option(DemoOption(option_name, demos))
            return EvolveType.RANGE
        return EvolveType.ID


    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, max_num, current_best_score_sum = data['name'], data['module_name'], data['max_num'], data['current_best_score_sum']
        allow_duplicate = data.get('allow_duplicate', False)
        param = cls(
            name=name, 
            module_name=module_name, 
            max_num=max_num,
            allow_duplicate=allow_duplicate,
        )
        
        demo_cache = {}
        for e in data['demo_cache']:
            demo = Demonstration(**e)
            demo_cache[demo.id] = demo
        param.demo_cache = demo_cache
        param.best_score_by_task = data['best_score_by_task']
        
        demo_pq = [
            (
                e['score'], 
                e['task_id'], 
                e['demo_id'],
            ) for e in data['demo_pq']
        ]
        heapq.heapify(demo_pq) 
        param.demo_pq = demo_pq
        
        task_id_set = set(data['task_id_set'])
        param.task_id_set = task_id_set
        
        loaded_options = data['options']
        loaded_options.pop('Identity', None)
        loaded_options = {name: DemoOption.from_dict(option, demo_cache) for name, option in loaded_options.items()}
        
        param.current_best_score_sum = current_best_score_sum
        param.options.update(loaded_options)
        return param
            
    
    def to_dict(self):
        base = super().to_dict()
        base['demo_cache'] = [dataclasses.asdict(v) for k, v in self.demo_cache.items()]
        base['best_score_by_task'] = self.best_score_by_task
        base['demo_pq'] = [
            {
                'score': score,
                'task_id': task_id,
                'demo_id': demo_id
            } for score, task_id, demo_id in self.demo_pq
        ]
        base['task_id_set'] = list(self.task_id_set)
        base['max_num'] = self.max_num
        base['current_best_score_sum'] = self.current_best_score_sum
        base['allow_duplicate'] = self.allow_duplicate
        return base

    @classmethod
    def bootstrap(
        cls,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: list[str] = [],
        max_num: int = 5,
        target_modules: Optional[list[str]] = None,
        log_path: Optional[str] = None,
        allow_duplicate: bool = False,
    ):
        """Collect good demos for LLMs in a workflow
        
        if target_modules is provided, only collect demos for these modules
        """
        # if log_path exists, load from it
        if log_path is not None:
            if os.path.exists(log_path):
                logger.info(f'Loading from {log_path}')
                return load_params(log_path)
        
        eval_task = EvalTask(
            script_path=script_path,
            args=script_args,
            other_python_paths=[],
            all_params={},
            module_name_paths={},
            aggregated_proposals={},
        )
        eval_result = evaluator.evaluate(eval_task)
        params = []
        module_with_demo = set()
        for module_2_demo in eval_result.demos:
            module_with_demo.update(module_2_demo.keys())
        # filter target modules
        module_with_demo = module_with_demo & set(target_modules) if target_modules else module_with_demo
        
        for m_name in module_with_demo:
            params.append(
                cls(
                    name='fewshot_demo',
                    max_num=max_num,
                    module_name=m_name, 
                    eval_result=eval_result,
                    inherit=False,
                    allow_duplicate=allow_duplicate,
                )
            )
        # if log_path provided, save to it
        if log_path is not None:
            logger.info(f'Saving to {log_path}')
            dir = os.path.dirname(log_path)
            if dir:
                os.makedirs(dir, exist_ok=True)
            dump_params(params, log_path)
        return params

    def custom_clean(self):
        self.demo_cache.clear()
        self.best_score_by_task.clear()
        self.task_id_set.clear()
        self.demo_pq.clear()
        self.current_best_score_sum = float('-inf')
    
class DemoOption(OptionBase):
    def __init__(self, tag: str, demos: list[Demonstration]):
        super().__init__(tag)
        self.demos = demos
    
    def _get_cost_indicator(self):
        return len(self.demos) + 1
    
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