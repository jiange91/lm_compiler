import json
import copy
import toml
import sys
import os

from rouge_score import rouge_scorer
        
def load_api_key(toml_file_path):
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)
        
        
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
    return score_dict['ROUGEL_f1']

def compare_two_answer(gold: dict, pred: dict):
    scores = {}
    for k in gold.keys():
        if k not in pred:
            pred[k] = ''
        scores[k] = compute_rouge_scores(gold[k], pred[k])
    return scores