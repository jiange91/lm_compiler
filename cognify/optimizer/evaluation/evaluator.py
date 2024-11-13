import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
import time
import copy
import logging
from dataclasses import dataclass, field
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from abc import ABC, abstractmethod
import multiprocessing as mp
import textwrap
from tqdm import tqdm
from cognify.optimizer import (
    get_registered_opt_program_entry, 
    get_registered_opt_modules, 
    get_registered_opt_score_fn,
)

import logging

from cognify.graph.program import Workflow, Module
from cognify.llm import CogLM, Demonstration
from cognify.cog_hub.common import CogBase
from cognify.cog_hub.utils import build_param
from cognify.optimizer.plugin import OptimizerSchema
from cognify.optimizer.core.flow import TopDownInformation, ModuleTransformTrace

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
        ids: Sequence[str],
        scores: Sequence[float],
        prices: Sequence[float],
        exec_times: Sequence[float],
        total_eval_cost: float,
        reduced_score: Optional[float] = None,
        reduced_price: Optional[float] = None,
        demos: Optional[Sequence[TDemoInTrial]] = None,
        
        meta: Optional[dict] = None,
    ) -> None:
        self.ids = ids
        self.scores = scores
        self.prices = prices
        self.exec_times = exec_times
        self.total_eval_cost = total_eval_cost
        self.reduced_score = reduced_score
        self.reduced_price = reduced_price
        self.demos = demos
        self.meta = meta
    
    def __str__(self) -> str:
        return (
            f"EvalResult: score: {self.reduced_score}, "
            f"price: {self.reduced_price}, " 
            f"{len(self.scores)} samples, "
            f"eval cost: {self.total_eval_cost}, "
            f"avg exec time: {sum(self.exec_times) / len(self.exec_times)} s"
        )
    
    def to_dict(self):
        """return result stats
        
        meta and demos are not included
        """
        stats = {}
        stats['summary'] = {
            'reduced_score': self.reduced_score,
            'reduced_price': self.reduced_price,
            'total_eval_cost': self.total_eval_cost,
        }
        stats['detailed'] = []
        for id, score, price, exec_time in zip(self.ids, self.scores, self.prices, self.exec_times):
            stats['detailed'].append({
                'id': id,
                'score': score,
                'price': price,
                'exec_time': exec_time,
            })
        return stats

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            ids=[d['id'] for d in data['detailed']],
            scores=[d['score'] for d in data['detailed']],
            prices=[d['price'] for d in data['detailed']],
            exec_times=[d['exec_time'] for d in data['detailed']],
            total_eval_cost=data['summary']['total_eval_cost'],
            reduced_score=data['summary']['reduced_score'],
            reduced_price=data['summary']['reduced_price'],
        )
    

@dataclass
class EvalTask:
    """Define a task to evaluate the score of a workflow
    """
    # execution env
    script_path: str
    args: list[str] # cmd args to the script
    other_python_paths: list[str]
    
    # transformation meta
    all_params: dict[str, CogBase] # {param_hash: param}
    module_name_paths: dict[str, str]
    aggregated_proposals: dict[str, dict[str, list[tuple[str, str]]]] # {layer_name: {module_name: [(param, option)]}}
    trace_back: list[str] = field(default_factory=list)
    
    def __getstate__(self) -> object:
        state = copy.deepcopy(self.__dict__)
        state.pop('all_params')
        state['all_params_ser'] = {}
        param_hashes = self.all_params.keys()
        for hash in param_hashes:
            state['all_params_ser'][hash] = self.all_params[hash].to_dict()
        return state

    def __setstate__(self, state: dict) -> None:
        param_pool = state.pop('all_params_ser')
        self.__dict__.update(state)
        self.all_params = {}
        # restore params
        for hash, param_dict in param_pool.items():
            self.all_params[hash] = build_param(param_dict)
    
    def to_dict(self) -> dict:
        return self.__getstate__()

    @classmethod
    def from_dict(cls, state: dict) -> 'EvalTask':
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj

    def add_PYTHON_PATH(self, evaluator_path: str):
        dirs = [os.path.dirname(self.script_path), os.path.dirname(evaluator_path)]
        for dir in dirs:
            if dir not in sys.path:
                sys.path.append(dir)
        if self.other_python_paths is not None:
            for path in self.other_python_paths:
                if path not in sys.path:
                    sys.path.append(path)
                
    def replay_module_transformations(self, ms: list[Module]) -> dict[str, Module]:
        """Replay the module transformations
        """
                        
        module_pool = {m.name: m for m in ms}
        module_ttrace = ModuleTransformTrace({m.name: type(m) for m in ms})
        # collect all module names that will be transformed
        all_opt_target_name = set()
        for proposal in self.aggregated_proposals.values():
            all_opt_target_name.update(proposal.keys())
        
        for layer_name, proposal in self.aggregated_proposals.items():
            for module_name, l_param_option in proposal.items():
                module = module_pool[module_name]
                for param_name, option_name in l_param_option:
                    param_hash = CogBase.chash(module_name, param_name)
                    param = self.all_params[param_hash]
                    
                    new_module, new_mapping = param.apply_option(option_name, module)
                    
                    for old_name, new_name in new_mapping.items():
                        module_ttrace.add_mapping(old_name, new_name)
                    module_pool[new_module.name] = new_module
                    for next_opt_target in Module.all_with_predicate(
                        [new_module], lambda m: m.name in all_opt_target_name
                    ):
                        module_pool[next_opt_target.name] = next_opt_target
                    module = new_module
        
        # check if modules are transformed correctly
        # check equivalence of current name path and registered name path
        assert module_ttrace.eq_transform_path(self.module_name_paths), "Module transformation not consistent"
        
        module_ttrace.mflatten()
        new_modules_dict = {
            ori_name: module_pool[new_name] 
            for ori_name, new_name in module_ttrace.flattened_name_paths.items()
        }
        return new_modules_dict

    def get_program_schema(self, evaluator_path: str) -> OptimizerSchema:
        self.add_PYTHON_PATH(evaluator_path)
        sys.argv = [self.script_path] + self.args
        schema = OptimizerSchema.capture(self.script_path, evaluator_path)
        
        logger.debug(f'opt_target_modules = {schema.opt_target_modules}')
        assert schema.opt_target_modules, "No module to optimize"
        return schema
                    
    def evaluate_program(
        self, 
        evaluator_path,
        input,
        label,
        task_index,
        sema,
        q: mp.Queue,
    ):
        schema = self.get_program_schema(evaluator_path)
        module_pool = {m.name: m for m in schema.opt_target_modules}
        
        # replace module invoke with new module
        # this does not replace the model but only the invoke function
        if self.aggregated_proposals:
            module_pool = self.replay_module_transformations(schema.opt_target_modules)
            for m in schema.opt_target_modules:
                new_module = module_pool[m.name]
                if isinstance(new_module, Workflow):
                    new_module.compile()
                logger.debug(f'replace {m} with {new_module}')
                m.invoke = new_module.invoke
        
        # clear execution state
        for m in module_pool.values():
            m.reset()
        start_time = time.time()
        result = schema.program(input)
        end_time = time.time()
        score = schema.score_fn(label, result)
        
        # get price and demo of running the program
        price = 0.0
        lm_to_demo = {}
        for lm in Module.all_of_type(module_pool.values(), CogLM):
            price += lm.get_total_cost()
            demo = lm.get_last_step_as_demo()
            if demo is not None:
                lm_to_demo[lm.name] = demo
            
        q.put((task_index, result, score, price, lm_to_demo, end_time - start_time))
        sema.release()

    
    def show_opt_trace(self) -> str:
        trace_lines = []
        trace_lines.append("********** Detailed Optimization Trace **********\n")
        
        for layer, proposals in self.aggregated_proposals.items():
            trace_lines.append(f"========== Layer: {layer} ==========")

            
            for module_name, param_options in proposals.items():
                trace_lines.append(f"\n  >>> Module: {module_name} <<<")
                
                for param_name, option_name in param_options:
                    param_hash = CogBase.chash(module_name, param_name)
                    param = self.all_params[param_hash]
                    class_path = f"{param.__class__.__module__}.{param.__class__.__name__}"
                    trace_lines.append(f"\n    - Parameter: <{class_path}>")
                    trace_lines.append(f"      Applied Option: {option_name}")
                    # Get the description with indentation for each line
                    option_description = param.options[option_name].describe()
                    option_description = textwrap.dedent(option_description)
                    
                    indented_description = "\n".join([f"        {line}" for line in option_description.splitlines()])
                    trace_lines.append(f"      Transformation Details:\n{indented_description}")
            
            trace_lines.append("\n" + "=" * 50 + "\n")

        
        # Combine all trace lines into a single string
        trace_dump = "\n".join(trace_lines)
        return trace_dump
        
    @classmethod
    def from_top_down_info(cls, tdi: TopDownInformation):
        return cls(
            script_path=tdi.script_path,
            args=tdi.script_args,
            other_python_paths=tdi.other_python_paths,
            all_params=tdi.all_params,
            module_name_paths=tdi.module_ttrace.module_name_paths,
            aggregated_proposals=tdi.module_ttrace.aggregated_proposals,
            trace_back=tdi.trace_back,
        )
    
class GeneralEvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Union[EvalTask, TopDownInformation],
        **kwargs,
    ) -> EvaluationResult:
        ...

def _gen_pbar_desc(level, tb, score, price):
    indent = '---' * level + '>'
    return f'{indent} Evaluation in {tb} | (avg score: {score:.2f}, avg cost@1000: {price*1000:.2f} $)'

class EvaluatorPlugin(GeneralEvaluatorInterface):
    def __init__(
        self,
        evaluator_path: str,
        trainset: Optional[Iterable[Tuple[any,any]]], # list of input data and labels
        evalset: Optional[Iterable[Tuple[any,any]]], # list of input data and labels
        testset: Optional[Iterable[Tuple[any,any]]], # list of input data and labels
        n_parallel: int = 1,
        score_reducer: Callable = None,
        price_reducer: Callable = None,
    ):
        self.evaluator_path = evaluator_path
        self.dataset = {
            'train': [trainset, None if not trainset else list(range(len(trainset)))],
            'eval': [evalset, None if not evalset else list(range(len(evalset)))],
            'test': [testset, None if not testset else list(range(len(testset)))]
        }
        
        self.n_parallel = n_parallel
        self.score_reducer = score_reducer if score_reducer is not None else default_reduer
        self.price_reducer = price_reducer if price_reducer is not None else default_reduer
    
    def evaluate(
        self,
        task: EvalTask,
        show_process: bool = False,
        pbar_position: int = 0,
        hierarchy_level: int = 0,
        **kwargs,
    ):
        return self.get_score(
            mode='train', 
            task=task, 
            show_process=show_process, 
            pbar_position=pbar_position,
            hierarchy_level=hierarchy_level,
        )
        
    def get_score(
        self,
        mode: Literal['train', 'eval', 'test'],
        task: EvalTask,
        show_process: bool,
        pbar_position: int = 0,
        hierarchy_level: int = 0,
    ):
        task.add_PYTHON_PATH(self.evaluator_path)
        logger.debug(f'sys_path = {sys.path}')
        
        data, indices = self.dataset[mode]
        n_parallel = min(self.n_parallel, len(indices))
        
        # Task queue to limit the number of parallel tasks
        # avoid worker pool to avoid reusing the same process
        all_workers = []
        sema = mp.Semaphore(n_parallel)
        result_q = mp.Queue()
        
        for task_index, pair_idx in enumerate(indices):
            input, label = data[pair_idx]
            sema.acquire()
            worker = mp.Process(
                target=task.evaluate_program, 
                args=(
                    self.evaluator_path,
                    input, label, 
                    task_index, sema, result_q
                )
            )
            worker.start()
            all_workers.append(worker)
            
        results = []
        
        if show_process:
            opt_trace = ' | '.join(task.trace_back)
            total_score, total_cost = 0.0, 0.0
            with tqdm(
                total=len(indices),
                desc=_gen_pbar_desc(hierarchy_level, opt_trace, 0.0, 0.0),
                leave=False,
                position=pbar_position,
            ) as pbar:
                for i in range(len(indices)):
                    results.append(result_q.get())
                    _, _, score, price, _, _ = results[-1]
                    total_score += score
                    total_cost += price
                    pbar.update(1)
                    pbar.set_description(
                        _gen_pbar_desc(hierarchy_level, opt_trace, total_score / (i+1), total_cost / (i+1))
                    )
        else:
            for i in range(len(indices)):
                results.append(result_q.get())
            
        for worker in all_workers:
            worker.join()
        # re-order the results according to the task index
        results = sorted(results, key=lambda x: x[0])
            
        prices = []
        scores = []
        demos = []
        exec_times = []
        for tid, result, score, price, demo, exec_time in results:
            prices.append(price)
            scores.append(score)
            demos.append(demo)
            exec_times.append(exec_time)
        reduced_score = self.score_reducer(scores)
        reduced_price = self.price_reducer(prices)
        return EvaluationResult(
            ids=[f'{mode}_{i}' for i in indices],
            scores=scores,
            prices=prices,
            exec_times=exec_times,
            total_eval_cost=sum(prices),
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            demos=demos,
        )
    
    def down_sample(
        self,
        mode: Literal['train', 'eval', 'test'],
        sample_size: int,
        task: EvalTask,
        sample_mode: Literal['random', 'difficulty'],
        prob_convertor: Callable[[EvaluationResult], Sequence[int]] = None,
        log_dir: str = "eval_down_sample_logs",
    ):
        """Generate a subset of the dataset according to answer score
        
        The objective is to reduce the evaluation cost with the following two principles:
        
        1. subset should have good coverage of the input space, spanning from easy to hard
        2. harder questions are more important
        
        In case the task itself does not provide meaningful comparative scores (e.g. classification task), use `random` sample mode to randomly sample from the eval_set or use `difficulty` at your own risk.
        
        NOTE: since we only care about comparative score, maybe use the most efficient config with least bias (e.g. cheap models) to evaluate the subset
        
        The default prob_convertor works fine for score within range[0,1], but you can provide a custom one if needed
        
        also please be informed that we always assume score is higher the better
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{mode}_sub_ids.json')

        # validate
        full_indices = list(range(len(self.dataset[mode][0])))
        if sample_size > len(full_indices):
            raise ValueError(f'Sample size {sample_size} is larger than the full dataset size {len(full_indices)}')
        
        if os.path.exists(log_path):
            logger.info(f'Loading downsampled indices from {log_path}')
            indices = json.load(open(log_path, 'r'))
            if len(indices) != sample_size:
                raise ValueError(f'Loaded data size {len(indices)} does not match sample size {sample_size}')
            self.dataset[mode][1] = indices
            return
        
        if sample_mode == 'random':
            indices = np.random.choice(full_indices, size=sample_size, replace=False).tolist()
        else:
            logger.info('Down sampling with difficulty, start profiling...')
            dry_run_path = os.path.join(log_dir, f'dry_run_{mode}.json')
            if os.path.exists(dry_run_path):
                logger.info(f'Loading dry run results from {dry_run_path}')
                eval_result = EvaluationResult.from_dict(json.load(open(dry_run_path, 'r')))
            else:
                eval_result = self.get_score(mode, task, show_process=True)
                with open(dry_run_path, 'w+') as f:
                    json.dump(eval_result.to_dict(), f, indent=4)
            # if user provide a custom prob convertor
            if prob_convertor is not None:
                probs = prob_convertor(eval_result)
                indices = np.random.choice(full_indices, size=sample_size, replace=False, p=probs).tolist()
            else:
                # sampling prob is reverse to the score
                # also smooth it to reduce extremely easy or hard questions
                def transform(x):
                    return np.exp(-x)
                scaled_reverse_score = transform(np.array(eval_result.scores))
                # normalize to prob
                probs = scaled_reverse_score / scaled_reverse_score.sum()
                # sample according to the prob
                indices = np.random.choice(full_indices, size=sample_size, replace=False, p=probs).tolist()
                
        json.dump(sorted(indices), open(log_path, 'w'), indent=4)
        self.dataset[mode][1] = indices
    