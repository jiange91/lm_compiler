#================================================================
# Evaluator
#================================================================

from cognify.optimizer.evaluation.metric import F1Str

metric = F1Str()

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
        input = (d["question"], d["docs"])
        output = d["answer"]
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.optimizer import default_search

search_settings = default_search.create_search()