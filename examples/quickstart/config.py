#================================================================
# Evaluator
#================================================================

import cognify
from cognify.optimizer.registry import register_opt_score_fn

metric = cognify.metric.f1_score_str

@register_opt_score_fn
def evaluate_answer(answer, label):
    return metric(answer, label)

#================================================================
# Data Loader
#================================================================

from cognify.optimizer.registry import register_data_loader
import json

@register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = {
            'question': d["question"], 
            'documents': d["docs"]
        }
        output = {
            'label': d["answer"],
        }
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

search_settings = default.create_search(
    n_trials=5,
)