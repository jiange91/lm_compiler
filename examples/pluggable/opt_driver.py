from compiler.optimizer.layered_optimizer_pluggable import InnerLoopBayesianOptimization, OuterLoopOptimization
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluatorPlugin
import runpy
import uuid
import multiprocessing as mp

from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema

def opt():
    lm_options = [
        'gpt-4o-2024-08-06',
        'gpt-4o-mini',
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [ZeroShotCoT(), PlanBefore()]
    )
    
    few_shot_params = LMFewShot("few_shot", None, 2)
    
    scaffolding_params = LMScaffolding.bootstrap_from_source(
        script_path='/mnt/ssd4/lm_compiler/examples/pluggable/workload.py',
        decompose_threshold=0,
        log_dir='examples/pluggable/test_decompose_logs',
    )
    
    inner_loop = InnerLoopBayesianOptimization(
        universal_params=[model_param, few_shot_params, reasoning_param],
    )
    
    outer_loop = OuterLoopOptimization(
        dedicate_params=scaffolding_params,
    )
    
    input = {"task": "Write a single sentence pitch for our new product, be concise and impressive. Our new product is a smartwatch. It has a sleek design and is water resistant."}
    
    eval_set = [(input, None)] * 1
    evaluator = EvaluatorPlugin(
        eval_set=eval_set,
        n_parallel=3,
    )
    
    outer_loop.optimize(
        inner_loop=inner_loop,
        n_trials=6,
        script_path='/mnt/ssd4/lm_compiler/examples/pluggable/workload.py',
        evaluator=evaluator,
        resource_ratio=1/3,
        log_dir=f'examples/pluggable/logs_{uuid.uuid4()}',
    )
    
    # inner_loop.optimize(
    #     script_path='/mnt/ssd4/lm_compiler/examples/pluggable/ir_workload.py',
    #     n_trials=4,
    #     evaluator=evaluator,
    #     log_dir=f'examples/pluggable/logs_{uuid.uuid4()}',
    #     throughput=1,
    # )
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    opt()