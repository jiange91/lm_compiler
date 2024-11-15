import json
import copy
import toml
import sys
import os
import functools

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

# NOTE: the total prince is always the accurate price regardless of the passed in hyperthetical_model_options
#       per_gpt_use shows the token consumed for each GPT, hypertheical if provided
#       h_price is the total price based on per_gpt_usage
def get_bill(token_usage, hyperthetical_model_options = None):
    def model_2_price_pM(model: str, prompt, completion):
        if 'gpt-4o-mini' in model:
            return (0.15 * prompt +  0.6 * completion) / 1e6
        elif 'gpt-4o-2024-05-13' in model:
            return (5 * prompt + 15 * completion) / 1e6
        elif 'gpt-4o-2024-08-06' in model:
            return (2.5 * prompt + 10 * completion) / 1e6
    per_gpt_use = {}
    for step, usage in token_usage.items():
        if step == 'total':
            total_price = 0
            for model, usage in usage.items():
                total_price += model_2_price_pM(model, usage['prompt_tokens'], usage['completion_tokens'])
        else:
            for model, usage in usage.items():
                if hyperthetical_model_options is not None:
                    model = hyperthetical_model_options[step]
                if model not in per_gpt_use:
                    per_gpt_use[model] = [0, 0]
                per_gpt_use[model][0] += usage['prompt_tokens']
                per_gpt_use[model][1] += usage['completion_tokens']
    h_price = 0
    for model, usage in per_gpt_use.items():
        price = model_2_price_pM(model, usage[0], usage[1])
        h_price += price
    return total_price, per_gpt_use, h_price

            
def deprecate_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        raise DeprecationWarning(f"{func.__name__} is deprecated")
    return wrapper
