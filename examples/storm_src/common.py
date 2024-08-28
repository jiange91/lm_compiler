from collections import defaultdict
from compiler.IR.program import StatePool
from rouge_score import rouge_scorer
from evaluate import load

# Define State
class StormState(StatePool):
    def __init__(self):
        super().__init__()
    
    def dump(self, path: str):
        pass
    
    def load(self, path: str):
        pass

def compute_rouge_scores(golden_answer: str, predicted_answer: str):
    """
    Compute rouge score for given output and golden answer to compare text overlap.
        - golden_answer: plain text of golden answer
        - predicted_answer: plain text of predicted answer
    """

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(golden_answer, predicted_answer)
    score_dict = {}
    for metric, metric_score in scores.items():
        score_dict[f'{metric.upper()}_precision'] = metric_score.precision
        score_dict[f'{metric.upper()}_recall'] = metric_score.recall
        score_dict[f'{metric.upper()}_f1'] = metric_score.fmeasure
    return score_dict

def compare_two_answer(gold: dict, pred: dict):
    scores = {}
    for k in gold.keys():
        if k not in pred:
            pred[k] = ''
        scores[k] = compute_rouge_scores(gold[k], pred[k])['ROUGEL_f1']
    return scores

# Load BERTScore
print("Loading BERTScore...")
bertscore = load("bertscore")
print("BERTScore loaded.")
def bertscores(golden_answer: str, predicted_answer: str):
    results = bertscore.compute(predictions=[golden_answer], references=[predicted_answer], lang='en')
    return results
