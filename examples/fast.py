from compiler.dspy_bridge.interface import DSPyLM
from compiler.IR.program import Workflow, Module, StatePool
import dspy

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
print(input_state.state)
