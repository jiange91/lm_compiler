#================================================================
# Evaluator
#================================================================

import cognify

from pydantic import BaseModel

class Assessment(BaseModel):
    score: int
    
evaluator_prompt = """
You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

You should not solve the problem by yourself, a standard solution will be provided. 

Please rate the answer with a score between 0 and 10.
"""

evaluator_agent = cognify.StructuredModel(
    agent_name='llm_judge',
    system_prompt=evaluator_prompt,
    input_variables=(
        cognify.Input('problem'),
        cognify.Input('solution'),
        cognify.Input('answer'),
    ),
    output_format=cognify.OutputFormat(schema=Assessment),
    lm_config=cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0}),
    opt_register=False,
)

@cognify.register_evaluator
def llm_judge(problem, answer, solution):
    assess = evaluator_agent(inputs={'problem': problem, 'solution': solution, 'answer': answer})
    return assess.score


#================================================================
# Data Loader
#================================================================

import json
import random

@cognify.register_data_loader
def load_data():
    with open("data._json", "r") as f:
        data = json.load(f)
        
    random.seed(42)
    random.shuffle(data) 
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = {
            'problem': d["problem"],
        }
        ground_truth = {
            'solution': d["solution"],
        }
        new_data.append((input, ground_truth))
    return new_data[:30], None, new_data[30:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

model_configs = [
    # OpenAI models
    cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
    cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
]

search_settings = default.create_search(
    model_selection_cog=model_configs
)