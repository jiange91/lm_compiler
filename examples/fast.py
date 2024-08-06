from compiler.dspy_bridge.interface import DSPyLM
from compiler.IR.program import Workflow, Module, StatePool
from compiler.optimizer.bootstrap import BootStrapLMSelection
from storm_src.common import compare_two_answer

import dspy
import os

openai_kwargs = {
    'api_key': "sk-proj-YDNgNfy7Q1YfrjDBBau0T3BlbkFJIZmj9MMjGoGvvwNuMmE6",
    'api_provider': "openai",
    'temperature': 1.0,
    'top_p': 0.9,
}

class Plan(dspy.Signature):
    """Plan the subtask to perform given a query"""
    query = dspy.InputField()
    subtasks = dspy.OutputField()

def planner(query: str):
    predictor = dspy.Predict(Plan)
    subtasks = predictor(query=query).subtasks
    return {'subtasks': subtasks}

plan_module = DSPyLM('plan subtasks', planner)

class Worker(dspy.Signature):
    """Worker to perform subtasks"""
    
    subtasks = dspy.InputField()
    output = dspy.OutputField()

def worker(subtasks: str):
    worker = dspy.Predict(Worker)
    output = worker(subtasks=subtasks).output
    return {'output': output}

worker_module = DSPyLM('worker', worker)

workflow = Workflow()
workflow.add_module(plan_module)
workflow.add_module(worker_module)

workflow.add_edge(plan_module, worker_module)
workflow.set_exit_point(worker_module, 'output')

plan_module.lm_config = {'model': 'gpt-4o-mini', 'max_tokens': 500, **openai_kwargs}
worker_module.lm_config = {'model': 'gpt-4o-mini', 'max_tokens': 500, **openai_kwargs}

input_state = StatePool()
input_state.publish({'query': 'I want to write an article on the topic of AI'})

workflow.run(state=input_state)

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

def final_output_metric(gold: dict, pred: dict):
    gold['final_output'] = gold['final_output']
    pred['final_output'] = pred['final_output']
    return compare_two_answer(gold, pred)

trainset_input: list[StatePool] = [input_state]
trainset_label: list[str] = ["I want to write an article on the topic of AI"]

ms_boot = BootStrapLMSelection(
    workflow=workflow,
    teachers='gpt-4o',
    module_2_options=lm_options,
    module_2_metric=compare_two_answer,
    final_output_metric=final_output_metric,
    trainset_input=trainset_input,
    trainset_label=trainset_label,
    max_sample_to_keep=4,
)

common_dir = 'compile_log_fast/'
os.makedirs(common_dir, exist_ok=True)
label_path = os.path.join(common_dir, 'labels-4o.json')
profile_path = os.path.join(common_dir, 'module_option_profile.json')
curve_path = os.path.join(common_dir, 'storm_curve.joblib')
solution_path = os.path.join(common_dir, 'solutions_95.json')

solutions = ms_boot.compile(
    label_path=label_path,
    profile_path=profile_path,
    curve_path=curve_path,
    solution_path=solution_path,
)
