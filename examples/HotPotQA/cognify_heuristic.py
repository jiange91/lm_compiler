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
from compiler.optimizer.params import ensemble
from compiler.optimizer.core import driver, flow
import dspy
from dspy.datasets.hotpotqa import HotPotQA
from cognify_anno import first_query_agent, following_query_agent, answer_agent

def load_data_minor():
    trainset = [
        ("Are both Cangzhou and Qionghai in the Hebei province of China?", "no"),
        ("Who conducts the draft in which Marc-Andre Fleury was drafted to the Vegas Golden Knights for the 2017-18 season?", 'National Hockey League'),
        ("The Wings entered a new era, following the retirement of which Canadian retired professional ice hockey player and current general manager of the Tampa Bay Lightning of the National Hockey League (NHL)?", "Steve Yzerman"),
        ("What river is near the Crichton Collegiate Church?", "the River Tyne"),
        ("What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?", "design their own interdisciplinary program"),
        ("Which documentary was released first, Grizzly Man or Best Boy?", "Best Boy")
    ]
    return trainset[1:2], trainset[-1:]

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
    #================= LM Selection =================
    query_gen_lm_options = [
        LMConfig(
            # provider='fireworks',
            # kwargs= {
            #     'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
            # }
            provider='openai',
            kwargs= {
                'model': 'gpt-4o-mini',
            }
        )
    ]
    query_gen_model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(query_gen_lm_options)
    )
    query_gen_model_param.module_name = 'generate_query'
    
    refine_lm_options = [
        LMConfig(
            provider='openai',
            kwargs= {
                'model': 'gpt-4o-mini',
            }
        )
    ]
    refine_query_model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(refine_lm_options)
    )
    refine_query_model_param.module_name = 'refine_query'
    
    answer_lm_options = [
        LMConfig(
            provider='openai',
            kwargs= {
                'model': 'gpt-4o-mini',
            }
        )
    ]
    answer_query_model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(answer_lm_options)
    )
    answer_query_model_param.module_name = 'generate_answer'
    
    #================= Reasoning =================
    refine_reasoning_param = reasoning.LMReasoning(
        "reasoning", [ZeroShotCoT()] 
    )
    refine_reasoning_param.module_name = 'refine_query'
    answer_reasoning_param = reasoning.LMReasoning(
        "reasoning", [ZeroShotCoT()] 
    )
    answer_reasoning_param.module_name = 'generate_answer'
    
    #================= Few Shot =================
    # TOADD
    
    #================= Ensemble =================
    refine_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    refine_ensemble_param = ensemble.ModuleEnsemble(
        "ensemble", [refine_ensemble]
    )
    refine_ensemble_param.module_name = 'refine_query'
    answer_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    answer_ensemble_param = ensemble.ModuleEnsemble(
        "ensemble", [answer_ensemble]
    )
    answer_ensemble_param.module_name= 'generate_answer'
    
    #================= Evaluator =================
    evaluator = EvaluatorPlugin(
        eval_set=data,
        n_parallel=10,
    )
    
    #================= Inner Opt =================
    inner_opt_config = flow.OptConfig(
        n_trials=1,
        throughput=1,
        log_dir=None,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name='inner_loop',
        dedicate_params=[
            query_gen_model_param,
            refine_query_model_param,
            answer_query_model_param,
            
            refine_reasoning_param,
            answer_reasoning_param,
        ],
        opt_config=inner_opt_config,
        save_ckpt_interval=1,
    )
    
    #================= Outer Opt =================
    outer_opt_config = flow.OptConfig(
        n_trials=0,
        throughput=1,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/heuristic',
    )
    outer_loop_config = driver.LayerConfig(
        layer_name='outer_loop',
        dedicate_params=[
            refine_ensemble_param,
            answer_ensemble_param,
        ],
        opt_config=outer_opt_config,
        save_ckpt_interval=1,
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
        n_parallel=50,
    )
    eval_result = opt_driver.evaluate(
        evaluator=evaluator,
        bot_trial_log_id='0171a75bac4540bc99c793ca6be763c4',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/heuristic/inner_loop/05e247c93b1c4cc18d1e80b8298ebbb4/opt_logs.json',
    )
    print(eval_result)
    

    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    # train, val, dev = load_data()
    train, dev = load_data_minor()
    opt_driver = opt(train)
    eval(opt_driver, dev)
    