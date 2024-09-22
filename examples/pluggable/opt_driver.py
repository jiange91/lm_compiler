from compiler.optimizer.layered_optimizer_pluggable import InnerLoopBayesianOptimization
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluatorPlugin
import uuid
import multiprocessing as mp

def opt():
    lm_options = [
        'gpt-4o-2024-08-06',
        'gpt-4o-mini',
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    inner_loop = InnerLoopBayesianOptimization(
        universal_params=[model_param],
    )
    
    input = {"task": "Write a single sentence pitch for our new product, be concise and impressive. Our new product is a smartwatch. It has a sleek design and is water resistant."}
    
    eval_set = [(input, None)] * 1
    evaluator = EvaluatorPlugin(
        eval_set=eval_set,
        n_parallel=3,
    )
    
    inner_loop.optimize(
        script_path='/mnt/ssd4/lm_compiler/examples/pluggable/workload.py',
        n_trials=6,
        evaluator=evaluator,
        log_dir=f'examples/pluggable/logs_{uuid.uuid4()}',
        throughput=2,
    )

if __name__ == '__main__':
    mp.set_start_method('spawn')
    opt()