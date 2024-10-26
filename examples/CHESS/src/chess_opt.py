import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna
import argparse
from datetime import datetime
from typing import Any, Dict, List, TypedDict, Callable

from runner.task import Task

from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.params import ensemble
from compiler.IR.llm import LMConfig
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema
from compiler.optimizer.core import driver, flow
import dspy
from dspy.datasets.hotpotqa import HotPotQA

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline with the specified configuration."
    )
    parser.add_argument(
        "--data_mode", type=str, required=True, help="Mode of the data to be processed."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file."
    )
    parser.add_argument(
        "--pipeline_nodes",
        type=str,
        required=True,
        help="Pipeline nodes configuration.",
    )
    parser.add_argument(
        "--pipeline_setup",
        type=str,
        required=True,
        help="Pipeline setup in JSON format.",
    )
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="Flag to use checkpointing."
    )
    parser.add_argument(
        "--checkpoint_nodes",
        type=str,
        required=False,
        help="Checkpoint nodes configuration.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=False, help="Directory for checkpoints."
    )
    parser.add_argument(
        "--log_level", type=str, default="warning", help="Logging level."
    )
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.use_checkpoint:
        print("Using checkpoint")
        if not args.checkpoint_nodes:
            raise ValueError("Please provide the checkpoint nodes to use checkpoint")
        if not args.checkpoint_dir:
            raise ValueError("Please provide the checkpoint path to use checkpoint")

    return args

def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, "r") as file:
        dataset = json.load(file)
    return dataset[:]

def load_data(args):
    dataset = load_dataset(args.data_path)
    inputs = []
    for data in dataset:
        task = Task(data)
        result_dir = f"cognify_results/all_dev_manual_cot_demo/{task.db_id}/{task.question_id}/{args.run_start_time}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        inputs.append(
            {
                'args': args,
                'dataset': [data],
                'result_directory': result_dir,
            }
        )
    eval_data = [(input, None) for input in inputs]
    return eval_data[:5], eval_data[5:7], eval_data[7:10]

def opt(train, val, dev):
    evaluator = EvaluatorPlugin(
        trainset=train,
        # evalset=val,
        evalset=None,
        testset=dev,
        n_parallel=20,
    )
    # ================= LM Selection =================
    lm_options = [
        # LMConfig(
        #     provider='fireworks',
        #     cost_indicator=0.3,
        #     kwargs= {
        #         'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
        #         # 'temperature': 0.0,
        #     }
        # ),
        # LMConfig(
        #     provider='fireworks',
        #     model="accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-46c7347d",
        #     cost_indicator=0.6,
        #     kwargs= {
        #         'temperature': 0.0,
        #     }
        # ),
        # LMConfig(
        #     provider='local',
        #     model='llama-3.1-8b',
        #     cost_indicator=0.6,
        #     kwargs={
        #         'temperature': 0.0,
        #         'openai_api_base': 'http://192.168.1.16:30000/v1',
        #     }
        # ),
        LMConfig(
            provider='openai',
            model='gpt-4o-mini',
            cost_indicator=1.0,
            kwargs= {
                'temperature': 0.0,
            }
        )
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    # ================= Down Sample =================
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator.down_sample(
        sample_size=50,
        mode='train',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
    )
    evaluator.down_sample(
        sample_size=25,
        mode='eval',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
    )
    # ================= Sensitivity Analysis =================
    # model_sensitivity = SensitivityAnalyzer(
    #     target_param_type=model_selection.LMSelection,
    #     eval_task=plain_task,
    #     evaluator=evaluator,
    #     n_parallel=4,
    #     log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/sensitivity_logs',
    #     try_options=model_param,
    #     module_type=LangChainLM,
    # )
    # sensitivity_result = model_sensitivity.run()
    # print(sensitivity_result)
    
    # ================= Reasoning Options =================
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [IdentityOption(), ZeroShotCoT()] 
    )
    # ================= Few Shot Options =================
    few_shot_params = LMFewShot("few_shot", 4)
    # ================= Ensemble Options =================
    general_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    general_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [IdentityOption(), general_usc_ensemble]
    )
    
    refine_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    refine_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [refine_usc_ensemble]
    )
    refine_ensemble_params.module_name = 'refine_query'
    
    gen_answer_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    gen_answer_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [gen_answer_usc_ensemble]
    )
    gen_answer_ensemble_params.module_name = 'generate_answer'
    
    # ================= Inner Loop Config =================
    inner_opt_config = flow.OptConfig(
        n_trials=10,
        throughput=2,
        log_dir="/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_no_outer_fix_prompt_no_frugal/",
        evolve_interval=4,
        frugal_eval_cost=True,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name='inner_loop',
        universal_params=[few_shot_params, reasoning_param, model_param],
        opt_config=inner_opt_config,
        save_ckpt_interval=1,
    )
    
    outer_opt_config = flow.OptConfig(
        n_trials=0,
        throughput=4,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_full_opt',
        frugal_eval_cost=False,
    )
    
    outer_loop_config = driver.LayerConfig(
        layer_name='outer_loop',
        universal_params=[general_ensemble_params], # will overwrite module name
        # dedicate_params=[refine_ensemble_params, gen_answer_ensemble_params],
        opt_config=outer_opt_config,
        save_ckpt_interval=1,
        use_SH_allocation=True,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=[inner_loop_config],
        # layer_configs=[outer_loop_config, inner_loop_config],
        quality_constraint=0.52,
    )
    cost, pareto_frontier, opt_logs = opt_driver.run(
        evaluator=evaluator,
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    )
    return opt_driver

def eval(opt_driver: driver.MultiLayerOptimizationDriver):
    eval_result = opt_driver.evaluate(
        bot_trial_log_id='0801a67cbc474b93aaef22b8ca9b1587',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_no_outer_fix_prompt_no_frugal/opt_logs.json',
    )
    print(eval_result)

def raw_test(data):
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=data,
        n_parallel=100,
    )
    eval_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    print(evaluator.get_score('test', eval_task, show_process=True))

    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    train, val, dev = load_data()
    opt_driver = opt(train, val, dev)
    # eval(opt_driver)
    # raw_test(dev)