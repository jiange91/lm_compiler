#================================================================
# Evaluator
#================================================================

import re
import string
import unicodedata

def normalize_text(s):
    # Normalize Unicode characters
    s = unicodedata.normalize('NFD', s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Remove articles (a, an, the)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Fix extra whitespaces
    s = ' '.join(s.split())
    return s

from cognify.optimizer import register_opt_score_fn

@register_opt_score_fn
def f1_score_strings(label, pred):
    # Tokenize the strings (split by whitespace)
    tokens1 = set(normalize_text(label).split())
    tokens2 = set(normalize_text(pred).split())

    # Calculate true positives, false positives, and false negatives
    true_positives = len(tokens1 & tokens2)
    false_positives = len(tokens2 - tokens1)
    false_negatives = len(tokens1 - tokens2)

    if true_positives == 0:
        return 0
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


#================================================================
# Data Loader
#================================================================

from cognify.optimizer.registry import register_data_loader
import json

@register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
    # raw format:
    # [
    #     {
    #         "question": "...",
    #         "docs": [...],
    #         "answer": "..."
    #     },
    #     ...
    # ]
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = (d["question"], d["docs"])
        output = d["answer"]
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]

#================================================================
# Optimizer Set Up
#================================================================
from cognify.optimizer.control_param import ControlParameter
from cognify.cog_hub.optim_setup.qa import QASetup

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_setup=QASetup(),
    opt_history_log_dir='opt_results',
    evaluator_batch_size=2,
)