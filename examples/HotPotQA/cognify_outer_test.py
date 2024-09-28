from compiler.optimizer.layered_optimizer_pluggable import InnerLoopBayesianOptimization, OuterLoopOptimization
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask
import runpy
import uuid
import multiprocessing as mp
import json
import os
import random
import optuna

from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema
import dspy
from dspy.datasets.hotpotqa import HotPotQA

def load_data():
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    def get_input_label(x):
        return x.question, x.answer
    trainset = [get_input_label(x) for x in dataset.train[0:50]]
    devset = [get_input_label(x) for x in dataset.dev]
    print(devset[0])
    return trainset, devset[:50]

def load_data_minor():
    trainset = [
        ("Are both Cangzhou and Qionghai in the Hebei province of China?", "no"),
        ("Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?", 'National Hockey League'),
        ("The Wings entered a new era, following the retirement of which Canadian retired professional ice hockey player and current general manager of the Tampa Bay Lightning of the National Hockey League (NHL)?", "Steve Yzerman"),
        ("What river is near the Crichton Collegiate Church?", "the River Tyne"),
        ("What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?", "design their own interdisciplinary program"),
    ]
    return trainset, trainset[-1:]

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
    
    scaffolding_params = LMScaffolding.bootstrap_from_source(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        decompose_threshold=3,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/decompose_logs',
        default_identity=False,
    )
    
    few_shot_params = LMFewShot("few_shot", None, 4)
    
    inner_loop = InnerLoopBayesianOptimization(
        # universal_params=[model_param, few_shot_params, reasoning_param],
        universal_params=[few_shot_params, reasoning_param],
        # universal_params=[few_shot_params],
        save_ckpt_interval=1,
    )
    
    outer_loop = OuterLoopOptimization(
        dedicate_params=scaffolding_params,
        quality_constraint=0.4,
        save_ckpt_interval=1,
    )
    
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=20,
    )
    
    # cost, pareto_frontier = inner_loop.optimize(
    #     script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    #     n_trials=5,
    #     evaluator=evaluator,
    #     log_dir=f'/mnt/ssd4/lm_compiler/examples/HotPotQA/test_inner_log',
    #     throughput=2,
    # )
    cost, pareto_frontier = outer_loop.optimize(
        inner_loop=inner_loop,
        n_trials=20,
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        evaluator=evaluator,
        resource_ratio=1/10,
        log_dir=f'/mnt/ssd4/lm_compiler/examples/HotPotQA/debug_decomp_perf_logs',
        inner_throughput=1,
    )
    return pareto_frontier

def eval(data, config: InnerLoopBayesianOptimization.TrialLog):
    print(f"----- Testing trial {config.id} -----")
    trial, task = config.program
    print("  Params: {}".format(trial.params))
    f1, f2 = trial.values
    print("  Values: score= {}, cost= {}".format(f1, f2))
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=50,
    )
    eval_result = evaluator.evaluate(task)
    print(str(eval_result))
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # train, dev = load_data_minor()
    train, dev = load_data()
    configs = opt(train)
    eval(dev, configs[1])
    