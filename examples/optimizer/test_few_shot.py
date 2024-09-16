import sys
import os
import json
import random


# Get the directory where the current file is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go to the parent directory or the directory you need to add
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from workload import qa_flow

from compiler.IR.program import StatePool
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput

class DummyMetric(MetricBase):
    move = MInput(str, "answer")
    def score(self, label, move):
        return random.uniform(0.0, 1.0)

state = StatePool()
state.init({
    "input": "1. d4 d5 2. Nf3 Nf6 3. e3 a6 4. Nc3 e6 5. Bd3 h6 6. e4 dxe4 7. Bxe4 Nxe4 8. Nxe4 Bb4+ 9. c3 Ba5 10. Qa4+ Nc6 11. Ne5 Qd5 12. f3 O-O 13. Nxc6 bxc6 14. Bf4 Ra7 15. Qb3 Qb5 16. Qxb5 cxb5 17. a4 bxa4 18. Rxa4 Bb6 19. Kf2 Bd7 20. Ke3 Bxa4 21. Ra1 Bc2 22. c4 Bxe4 23. fxe4 c5 24. d5 exd5 25. exd5 Re8+ 26. Kf3 Rae7 27. Rxa6 Bc7 28. Bd2 Re2 29. Bc3 R8e3+ 30. Kg4 Rxg2+ 31. Kf5", 
})
eval_set = [(state, None)]

evaluator = Evaluator(
    metric=DummyMetric(),
    eval_set=eval_set,
    num_thread=3,
)

lm_few_shot_params = LMFewShot.bootstrap(qa_flow, evaluator, 2, log_path='test_fs.json')

from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.params.meta_programming.mp import MetaPrompting
from compiler.optimizer.layered_optimizer import InnerLoopBayesianOptimization
from compiler.optimizer.params.utils import load_params

def opt():
    lm_options = [
        'gpt-4o',
        'gpt-4o-mini',
    ]
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [common.IdentityOption(), ZeroShotCoT(), PlanBefore()]
    )

    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    inner_loop = InnerLoopBayesianOptimization(
        dedicate_params=lm_few_shot_params,
        universal_params=[model_param, reasoning_param],
    )

    inner_loop.optimize(
        workflow=qa_flow, 
        evaluator=evaluator, 
        n_trials=10,
        throughput=4,
        log_dir='examples/optimizer/test_logs',
    )
    
opt()

# def load_opt():
#     params = load_params('examples/optimizer/test_logs/inner_loop/params.json')
#     inner_loop = InnerLoopBayesianOptimization(
#         dedicate_params=params,
#     )
#     inner_loop.optimize(
#         workflow=qa_flow, 
#         evaluator=evaluator, 
#         n_trials=1,
#         throughput=1,
#         log_dir='examples/optimizer/resumed_test_logs',
#     )
    
# load_opt()