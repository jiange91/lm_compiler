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

from cognify.llm.model import LMConfig

# Define model configurations, each encapsulated in a ModelOption
model_configs = [
    # OpenAI model
    LMConfig(
        custom_llm_provider='openai',
        model='gpt-4o-mini',
        cost_indicator=1.0,
        kwargs={'temperature': 0.0}
    ),
    LMConfig(
        custom_llm_provider='openai',
        model='gpt-4o',
        cost_indicator=10.0,
        kwargs={'temperature': 0.0}
    ),
]

from cognify.cog_hub import default_search

search_settings = default_search.create_search(
    opt_log_dir='try_new_thing',
    model_selection_cog=model_configs,
)