from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.params import ensemble
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
from compiler.optimizer.core import driver, flow
import dspy
from dspy.datasets.hotpotqa import HotPotQA

def load_data_minor():
    trainset = [
        ("Are both Cangzhou and Qionghai in the Hebei province of China?", "no"),
        ("Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?", 'National Hockey League'),
        ("The Wings entered a new era, following the retirement of which Canadian retired professional ice hockey player and current general manager of the Tampa Bay Lightning of the National Hockey League (NHL)?", "Steve Yzerman"),
        ("What river is near the Crichton Collegiate Church?", "the River Tyne"),
        ("What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?", "design their own interdisciplinary program"),
        ("Which documentary was released first, Grizzly Man or Best Boy?", "Best Boy")
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
        # LMConfig(
        #     provider='fireworks',
        #     cost_indicator=0.3,
        #     kwargs= {
        #         'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
        #         # 'temperature': 0.0,
        #     }
        # ),
        LMConfig(
            provider='openai',
            cost_indicator=1.0,
            kwargs= {
                'model': 'gpt-4o-mini',
                # 'temperature': 0.0,
            }
        )
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=25,
    )
    evaluator.down_sample(
        sample_size=25,
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
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
    
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [IdentityOption(), ZeroShotCoT()] 
    )
    few_shot_params = LMFewShot("few_shot", 2)
    inner_opt_config = flow.OptConfig(
        n_trials=16,
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
        throughput=1,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/more_control',
    )
    
    usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [IdentityOption(), usc_ensemble]
    )
    ensemble_params.module_name = 'generate_answer'
    outer_loop_config = driver.layerConfig(
        layer_name='outer_loop',
        # universal_params=[ensemble_params],
        dedicate_params=[ensemble_params],
        opt_config=outer_opt_config,
        save_ckpt_interval=1,
        # use_SH_allocation=True,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=[outer_loop_config, inner_loop_config],
    )
    cost, pareto_frontier, opt_logs = opt_driver.run(
        evaluator=evaluator,
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    )
    return opt_driver

def eval(opt_driver, data):
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=100,
    )
    eval_result = opt_driver.evaluate(
        evaluator=evaluator,
        bot_trial_log_id='4808fc6fb2db4d18a219caf88a8e51d7',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/more_control/inner_loop/4a853ee5754b4e39a8d1d650b0d94d2f/opt_logs.json',
    )
    print(eval_result)
    

    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    # train, val, dev = load_data()
    train, dev = load_data_minor()
    opt_driver = opt(train)
    eval(opt_driver, dev)
    