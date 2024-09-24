import json
import os
import uuid
import logging
import random

import dspy
import dspy.evaluate
from plot_module import MatPlotModule

def load_data():
    data_path = 'examples/dspy_matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    all_data = []
    for item in data:
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id'] 
        directory_path = f'examples/dspy_matplot_agent/sample_runs'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
        item = dspy.Example({
            'query': novice_instruction,
            "directory_path": directory_path,
            "example_id": example_id,
            "input_path": f'{data_path}/data/{example_id}',
            
            "ground_truth": f"/mnt/ssd4/lm_compiler/examples/dspy_matplot_agent/benchmark_data/ground_truth/example_{example_id}.png",
        }).with_inputs("query", "directory_path", "example_id", "input_path")
        all_data.append(item)
        
    return all_data

from evaluator import gpt_4v_evaluate 

def matplot_eval(gold, pred, trace=None) -> float:
    return gpt_4v_evaluate(gold.ground_truth, pred.img_path, pred.rollback)

all_data = load_data()
random.seed(2333)
random.shuffle(all_data)

train, test = all_data[:4], all_data[4:]
print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

evaluator = dspy.evaluate.Evaluate(
    devset=test,
    num_thread=10,
    display_progress=True, 
    display_table=1,
    metric=matplot_eval,
)

matplot_module = MatPlotModule(model_type='gpt-4o-mini')
# evaluator(matplot_module)
# exit()

from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
prmopt_model = dspy.OpenAI(model='gpt-4o-mini', max_tokens=1000)
dspy.settings.configure(lm=prmopt_model)
bootstrap_optimizer = MIPROv2(
    prompt_model=prmopt_model,
    metric=matplot_eval,
    num_candidates=4,
)


optimized_matplot = bootstrap_optimizer.compile(
    matplot_module,
    trainset=train,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    num_trials=15,
    minibatch=False,
)

optimized_matplot.save('examples/dspy_matplot_agent/optimized_matplot_module')
evaluator(optimized_matplot)