import sys
import os
import json
import random


# Get the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go to the parent directory or the directory you need to add
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from more_complex_workload import qa_flow

from compiler.IR.program import StatePool
from compiler.cog_hub.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput

class DummyMetric(MetricBase):
    move = MInput(str, "answer")
    def score(self, label, move):
        return random.uniform(0.0, 1.0)


from compiler.cog_hub.utils import dump_params, load_params
from compiler.cog_hub.scaffolding import LMScaffolding
from compiler.optimizer.layered_optimizer import InnerLoopBayesianOptimization
from compiler.cog_hub.utils import load_params
from compiler.optimizer.decompose import LMTaskDecompose

# scaffolding_params = LMScaffolding.bootstrap(
#     qa_flow,
#     decompose_threshold=0,
#     log_dir='examples/optimizer/test_decompose_logs',
# )

decomposer = LMTaskDecompose(
    workflow=qa_flow,
)

decomposer.decompose(
    # target_modules=['initial code generation', 'plot debugger'],
    log_dir='examples/optimizer/test_decompose_logs',
    threshold=0,
    materialize=True,
)

input = StatePool()
input.init({
    'question': "what is the capital of France?",
    "doc": 'Capital One Financial Corporation is an American bank holding company founded on July 21, 1994 and specializing in credit cards, auto loans, banking, and savings accounts, headquartered in Tysons, Virginia with operations primarily in the United States.',
})

qa_flow.pregel_run(input)
print(input.news('decision'))
# dump_params(scaffolding_params, 'examples/optimizer/scaffolding_params.json')

# loaded_params = load_params('examples/optimizer/scaffolding_params.json')
