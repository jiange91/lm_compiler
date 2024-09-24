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

def load_data():
    data_path = 'examples/IR_matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    all_data = []
    for item in data:
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id'] 
        directory_path = f'examples/IR_matplot_agent/sample_runs'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
        input = {
            'query': novice_instruction,
            "directory_path": directory_path,
            "example_id": example_id,
            "input_path": f'{data_path}/data/{example_id}',
        }
        label = {"ground_truth": f"/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/benchmark_data/ground_truth/example_{example_id}.png"}
        all_data.append((input, label))
    return all_data



def opt(train):
    lm_options = [
        'gpt-4o-2024-08-06',
        'gpt-4o-mini',
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [IdentityOption(), ZeroShotCoT(), PlanBefore()]
    )
    
    few_shot_params = LMFewShot("few_shot", None, 2)
    
    scaffolding_params = LMScaffolding(
        name='scaffolding',
        log_dir='examples/IR_matplot_agent/test_decompose_logs',
    )
    
    inner_loop = InnerLoopBayesianOptimization(
        universal_params=[few_shot_params, reasoning_param],
    )
    
    outer_loop = OuterLoopOptimization(
        universal_params=[scaffolding_params],
    )
    
    evaluator = EvaluatorPlugin(
        eval_set=train,
        n_parallel=5,
    )
    
    # outer_loop.optimize(
    #     inner_loop=inner_loop,
    #     n_trials=6,
    #     script_path='/mnt/ssd4/lm_compiler/examples/pluggable/workload.py',
    #     evaluator=evaluator,
    #     resource_ratio=1/3,
    #     log_dir=f'examples/pluggable/logs_{uuid.uuid4()}',
    # )
    
    cost, pareto_frontier = inner_loop.optimize(
        script_path='/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/workflow.py',
        n_trials=15,
        evaluator=evaluator,
        log_dir=f'examples/IR_matplot_agent/opt_logs',
        throughput=3,
    )
    return pareto_frontier

def eval(trial: optuna.trial.FrozenTrial, task: EvalTask, test):
    print("----- Testing select trial -----")
    print("  Params: {}".format(trial.params))
    f1, f2 = trial.values
    print("  Values: score= {}, cost= {}".format(f1, f2))
    
    evaluator = EvaluatorPlugin(
        eval_set=test,
        n_parallel=10,
    )
    eval_result = evaluator.evaluate(task)
    print(str(eval_result))

if __name__ == '__main__':
    all_data = load_data()
    train, test = all_data[:4], all_data[4:]
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    
    mp.set_start_method('spawn')
    
    best_trials = opt(train)
    eval(*best_trials[0], test)