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

from compiler.optimizer import register_opt_score_fn

@register_opt_score_fn
def f1(label: str, pred: str) -> float:
    score = f1_score_strings(label, pred)
    return score
