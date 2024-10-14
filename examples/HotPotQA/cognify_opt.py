from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from compiler.langchain_bridge.interface import LangChainLM
import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna

from compiler.IR.llm import LMConfig
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema
import dspy
from dspy.datasets.hotpotqa import HotPotQA

def load_data_minor():
    trainset = [
        ("Are both Cangzhou and Qionghai in the Hebei province of China?", "no"),
        ("Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?", 'National Hockey League'),
        ("The Wings entered a new era, following the retirement of which Canadian retired professional ice hockey player and current general manager of the Tampa Bay Lightning of the National Hockey League (NHL)?", "Steve Yzerman"),
        ("What river is near the Crichton Collegiate Church?", "the River Tyne"),
        ("What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?", "design their own interdisciplinary program"),
    ]
    return trainset, trainset[-1:]

def load_data():
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    def get_input_label(x):
        return x.question, x.answer
    trainset = [get_input_label(x) for x in dataset.train[0:100]]
    valset = [get_input_label(x) for x in dataset.train[100:150]]
    devset = [get_input_label(x) for x in dataset.dev]
    print(devset[0], len(devset))
    return trainset, valset, devset

def opt(data):
    lm_options = [
        LMConfig(
            provider='fireworks', 
            kwargs= {
                'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
                'temperature': 0.0,
            }
        ),
        LMConfig(
            provider='openai',
            kwargs= {
                'model': 'gpt-4o-mini',
                'temperature': 0.0,
            }
        )
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=32,
    )
    
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    
    model_sensitivity = SensitivityAnalyzer(
        target_param_type=model_selection.LMSelection,
        eval_task=plain_task,
        evaluator=evaluator,
        n_parallel=4,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/sensitivity_logs',
        try_options=model_param,
        module_type=LangChainLM,
    )
    sensitivity_result = model_sensitivity.run()
    print(sensitivity_result)
    return 

    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    # train, val, dev = load_data()
    train, dev = load_data_minor()
    configs = opt(train)
    