from cognify.optimizer import register_evaluator
from dsp.utils.metrics import HotPotF1, F1

@register_evaluator
def answer_f1(label: str, pred: str):
    if isinstance(label, str):
        label = [label]
    score = F1(pred, label)
    return score