from compiler.optimizer import register_opt_score_fn
from dsp.utils.metrics import HotPotF1, F1

@register_opt_score_fn
def answer_f1(label: str, pred: str):
    if isinstance(label, str):
        label = [label]
    score = F1(pred, label)
    return score