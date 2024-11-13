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

from cognify.optimizer import register_opt_score_fn

@register_opt_score_fn
def f1(label: str, pred: str) -> float:
    score = f1_score_strings(label, pred)
    return score

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

from cognify.optimizer.core import driver, flow
from cognify.cog_hub import reasoning, ensemble
from cognify.cog_hub.common import NoChange
from cognify.cog_hub.fewshot import LMFewShot
from cognify.cog_hub.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter

# ================= Inner Loop Config =================
# Reasoning Parameter
reasoning_param = reasoning.LMReasoning(
    [NoChange(), ZeroShotCoT()] 
)
# Few Shot Parameter
few_shot_params = LMFewShot(2)

# Layer Config
inner_opt_config = flow.OptConfig(
    n_trials=6,
)
inner_loop_config = driver.LayerConfig(
    layer_name='inner_loop',
    universal_params=[few_shot_params, reasoning_param],
    opt_config=inner_opt_config,
)

# ================= Outer Loop Config =================
# Ensemble Parameter
general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
general_ensemble_params = ensemble.ModuleEnsemble(
    [NoChange(), general_usc_ensemble]
)
# Layer Config
outer_opt_config = flow.OptConfig(
    n_trials=2,
)
outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    universal_params=[general_ensemble_params],
    opt_config=outer_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, inner_loop_config],
    opt_history_log_dir='opt_results',
)