import sys
import os
import json

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
        return 0.5

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

lm_few_shot_params = LMFewShot.bootstrap(qa_flow, evaluator, 2, 'test_fs.json')
