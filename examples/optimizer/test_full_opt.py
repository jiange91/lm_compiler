import sys
import os
import json
import random
import uuid


# Get the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go to the parent directory or the directory you need to add
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from more_complex_workload import qa_flow

from compiler.IR.program import StatePool
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput

class DummyMetric(MetricBase):
    decision = MInput(str, "answer")
    
    def score(self, label, decision):
        return random.uniform(0.0, 1.0)

state = StatePool()
state.init({
    'question': "abc",
    "doc": 'abc',
})
eval_set = [(state, None)]

evaluator = Evaluator(
    metric=DummyMetric(),
    eval_set=eval_set,
    num_thread=1,
)

from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.params.utils import dump_params, load_params
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.layered_optimizer import InnerLoopBayesianOptimization, OuterLoopOptimization
from compiler.optimizer.params.utils import load_params

scaffolding_params = LMScaffolding.bootstrap(
    qa_flow,
    decompose_threshold=0,
    log_dir='examples/optimizer/test_decompose_logs'
)

def opt():
    lm_options = [
        'gpt-4o',
        'gpt-4o-mini',
    ]

    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options),
        module_name='doc_filter',
        inherit=True,
    )
    
    inner_loop = InnerLoopBayesianOptimization(
        dedicate_params=[model_param],
    )
    
    outer_loop = OuterLoopOptimization(
        dedicate_params=scaffolding_params,
        quality_constraint=0.8,
    )
    
    outer_loop.optimize(
        qa_flow,
        inner_loop,
        evaluator,
        n_trials=20,
        resource_ratio=1/4,
        throughput=2,
        inner_throughput=2,
        log_dir=f'examples/optimizer/test_logs/{uuid.uuid4()}',
    )
    
opt()
