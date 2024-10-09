from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.params import reasoning, model_selection, common
import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna

from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.params import ensemble
from compiler.optimizer.plugin import OptimizerSchema
import dspy
from compiler.optimizer.evaluation.evaluator import EvaluatorPlugin, EvalTask
from dspy.datasets.hotpotqa import HotPotQA
from compiler.optimizer.core import driver, flow
from compiler.optimizer.params.fewshot import LMFewShot

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
    return trainset, valset, devset

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
    
    few_shot_params = LMFewShot("few_shot", 8)
    
    usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [IdentityOption(), usc_ensemble]
    )

    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=40,
    )
    
    down_sample_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator.down_sample(3, down_sample_task, 'difficulty')
    
    return
    
    inner_opt_config = flow.OptConfig(
        n_trials=8,
        throughput=2,
        log_dir=None,
    )
    inner_loop_config = driver.layerConfig(
        layer_name='inner_loop',
        universal_params=[few_shot_params, reasoning_param],
        opt_config=inner_opt_config,
        save_ckpt_interval=1,
    )
    
    outer_opt_config = flow.OptConfig(
        n_trials=0,
        throughput=2,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/whole_opt_test',
    )
    outer_loop_config = driver.layerConfig(
        layer_name='outer_loop',
        universal_params=[ensemble_params],
        opt_config=outer_opt_config,
        use_SH_allocation=True,
        save_ckpt_interval=1,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=[outer_loop_config, inner_loop_config],
        evaluator=evaluator,
        quality_constraint=0.4,
    )
    
    cost, pareto_frontier, opt_logs = opt_driver.run(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    )
    
    eval_result = opt_driver.evaluate(
        bot_trial_log_id='10255fbe478343c885dcfc566a734226',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/whole_opt_test/inner_loop/17dfd7780bc646b9b11dee72fec2f395/opt_logs.json',
    )
    print(eval_result)
    
    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    # train, val, dev = load_data()
    train, dev = load_data_minor()
    opt(train)