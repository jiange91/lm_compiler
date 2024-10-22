from compiler.optimizer import core
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.importance_eval import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask
import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna

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
    print(devset[0])
    return trainset, valset, devset[:50]

def opt(data):
    lm_options = [
        'gpt-4o-2024-08-06',
        'gpt-4o-mini',
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [IdentityOption(), ZeroShotCoT()] 
    )
    
    few_shot_params = LMFewShot("few_shot", None, 2)

    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=5,
    )
    
    inner_loop = core.BottomLevelOptimization(
        name='inner_loop',
        evaluator=evaluator,
        # universal_params=[model_param, few_shot_params, reasoning_param],
        universal_params=[few_shot_params, reasoning_param],
        # universal_params=[few_shot_params],
        save_ckpt_interval=1,
    )
    
    opt_config = core.OptConfig(
        n_trials=10,
        throughput=2,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/opt_test',
    )
    
    cost, pareto_frontier, opt_logs = inner_loop.easy_optimize(
        opt_config=opt_config,
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    )
    
    # eval_result = inner_loop.easy_eval(
    #     trial_log_id='10ae3766ecef4b2da2b79ef23947b14a',
    #     opt_config=opt_config,
    #     script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    # )
    # print(eval_result)
    
    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    # train, val, dev = load_data()
    train, dev = load_data_minor()
    opt(train)