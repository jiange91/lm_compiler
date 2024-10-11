import os
import sys
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Sequence
import copy
import logging
from dataclasses import dataclass
import optuna
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from abc import ABC, abstractmethod
import multiprocess as mp


import logging

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration, TokenUsageSummary, TokenUsage
from compiler.optimizer.evaluation.metric import MetricBase
from compiler.optimizer.params.common import ParamBase
from compiler.optimizer.plugin import OptimizerSchema
from compiler.utils import get_bill
from compiler.optimizer.core.flow import TopDownInformation, ModuleTransformTrace

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
        ids: Sequence[int],
        scores: Sequence[float],
        prices: Sequence[float],
        total_eval_cost: float,
        reduced_score: Optional[float] = None,
        reduced_price: Optional[float] = None,
        demos: Optional[Sequence[TDemoInTrial]] = None,
        
        meta: Optional[dict] = None,
    ) -> None:
        self.ids = ids
        self.scores = scores
        self.prices = prices
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
            f"eval cost: {self.total_eval_cost}"
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
    all_params: dict[str, ParamBase] # {param_hash: param}
    module_name_paths: dict[str, str]
    aggregated_proposals: dict[str, dict[str, list[tuple[str, str]]]] # {layer_name: {module_name: [(param, option)]}}
    
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
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
            t = ParamBase.registry[param_dict['type']]
            self.all_params[hash] = t.from_dict(param_dict)
    
    def to_dict(self) -> dict:
        return self.__getstate__()

    @classmethod
    def from_dict(cls, state: dict) -> 'EvalTask':
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj

    def add_PYTHON_PATH(self):
        dir = os.path.dirname(self.script_path)
        if dir not in sys.path:
            sys.path.append(dir)
        if self.other_python_paths is not None:
            for path in self.other_python_paths:
                if path not in sys.path:
                    sys.path.append(path)
                
    def replay_module_transformations(self, ms: list[Module]) -> dict[str, Module]:
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
                    param_hash = ParamBase.chash(module_name, param_name)
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

    def get_program_schema(self):
        self.add_PYTHON_PATH()
        sys.argv = [self.script_path] + self.args
        schema = OptimizerSchema.capture(self.script_path)
        
        logger.debug(f'opt_target_modules = {schema.opt_target_modules}')
        assert schema.opt_target_modules, "No module to optimize"
        return schema
                    
    def evaluate_program(self, input, label):
        schema = self.get_program_schema()
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
                m.reset()
            
        result = schema.program(input)
        score = schema.score_fn(label, result)
        
        # get price and demo of running the program
        usages = []
        lm_2_demo = {}
        for lm in Module.all_of_type(module_pool.values(), LLMPredictor):
            usages.extend(lm.get_token_usage())
            lm_2_demo[lm.name] = lm.get_step_as_example()
        
        summary = TokenUsageSummary.summarize(usages)
        price = summary.total_price
        return result, score, price, lm_2_demo
    
# class NoDaemonProcess(mp.Process):
#     @property
#     def daemon(self):
#         return False

#     @daemon.setter
#     def daemon(self, value):
#         pass


# class NoDaemonContext(type(mp.get_context(method="spawn"))):
#     Process = NoDaemonProcess

# # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# # because the latter is only a wrapper function, not a proper class.
# class MyPool(mp.pool.Pool):
#     def __init__(self, *args, **kwargs):
#         kwargs['context'] = NoDaemonContext()
#         super(MyPool, self).__init__(*args, **kwargs)

    @classmethod
    def from_top_down_info(cls, tdi: TopDownInformation):
        return cls(
            script_path=tdi.script_path,
            args=tdi.script_args,
            other_python_paths=tdi.other_python_paths,
            all_params=tdi.all_params,
            module_name_paths=tdi.module_ttrace.module_name_paths,
            aggregated_proposals=tdi.module_ttrace.aggregated_proposals,
        )
    
class GeneralEvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Union[EvalTask, TopDownInformation],
    ) -> EvaluationResult:
        ...

class EvaluatorPlugin(GeneralEvaluatorInterface):
    def __init__(
        self,
        eval_set: Iterable[Tuple[any,any]], # list of input data and labels
        n_parallel: int = 1,
        score_reducer: Callable = None,
        price_reducer: Callable = None,
    ):
        self.full_eval_set = eval_set
        self.eval_set = list(range(len(self.full_eval_set)))
        self.n_parallel = n_parallel
        self.score_reducer = score_reducer if score_reducer is not None else default_reduer
        self.price_reducer = price_reducer if price_reducer is not None else default_reduer
        
    def evaluate(
        self,
        task: EvalTask,
    ):
        task.add_PYTHON_PATH()
        logger.debug(f'sys_path = {sys.path}')
        n_parallel = min(self.n_parallel, len(self.eval_set)) 
        with mp.Pool(processes=n_parallel) as pool:
            tasks = []
            for pair_idx in self.eval_set:
                input, label = self.full_eval_set[pair_idx]
                tasks.append(
                    pool.apply_async(task.evaluate_program, args=(input, label))
                )
            results = [task.get() for task in tasks]
            
        prices = []
        scores = []
        demos = []
        for result, score, price, demo in results:
            prices.append(price)
            scores.append(score)
            demos.append(demo)
        reduced_score = self.score_reducer(scores)
        reduced_price = self.price_reducer(prices)
        return EvaluationResult(
            ids=self.eval_set,
            scores=scores,
            prices=prices,
            total_eval_cost=sum(prices),
            reduced_score=reduced_score,
            reduced_price=reduced_price,
            demos=demos,
        )
    
    def down_sample(
        self,
        sample_size: int,
        task: EvalTask,
        sample_mode: Literal['random', 'difficulty'],
        prob_convertor: Callable[[EvaluationResult], Sequence[int]] = None,
    ):
        """Generate a subset of the eval_set according to answer score
        
        The objective is to reduce the evaluation cost with the following two principles:
        
        1. subset should have good coverage of the input space, spanning from easy to hard
        2. harder questions are more important
        
        In case the task itself does not provide meaningful comparative scores (e.g. classification task), use `random` sample mode to randomly sample from the eval_set or use `difficulty` at your own risk.
        
        NOTE: since we only care about comparative score, maybe use the most efficient config with least bias (e.g. cheap models) to evaluate the subset
        
        The default prob_convertor works fine for score within range[0,1], but you can provide a custom one if needed
        
        also please be informed that we always assume score is higher the better
        """
        full_indices = list(range(len(self.full_eval_set))) 
        if sample_mode == 'random':
            self.eval_set = np.random.choice(full_indices, size=sample_size, replace=False).tolist()
            return

        eval_result = self.evaluate(task)
        # if user provide a custom prob convertor
        if prob_convertor is not None:
            probs = prob_convertor(eval_result)
            self.eval_set = np.random.choice(full_indices, size=sample_size, replace=False, p=probs).tolist()
            return
        
        # sampling prob is reverse to the score
        # also smooth it to reduce extremely easy or hard questions
        def transform(x):
            return np.exp(-x)
        scaled_reverse_score = transform(np.array(eval_result.scores))
        # normalize to prob
        probs = scaled_reverse_score / scaled_reverse_score.sum()
        # sample according to the prob
        self.eval_set = np.random.choice(full_indices, size=sample_size, replace=False, p=probs).tolist()
 
        
        