from compiler.utils import load_api_key, get_bill
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.output_parsers import StrOutputParser
import sys
import os
import random
import argparse
import operator
from typing import Annotated, List, Tuple, TypedDict
from copy import deepcopy
from typing import List
from compiler.langchain_bridge.interface import get_inspect_runnable

import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
    base
)
from langchain_openai.chat_models import ChatOpenAI
from planner import planner_anno as planner
from executor import exec_anno as executor
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn


#==============================================================================
# Define graph
#==============================================================================

class PlanExecute(TypedDict):
    task: str
    steps: List[str]
    response: str
    
def plan_step(state: PlanExecute):
    plan = planner.invoke({"task": state["task"]})
    return {"steps": plan.steps}

def execute_step(state: PlanExecute):
    steps = state["steps"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    response = executor.invoke({"task": state["task"], "steps": plan_str})
    return {"response": response}
 
from compiler.IR.program import Workflow, Input, Output, StatePool
from compiler.IR.modules import CodeBox

app = Workflow('qa_flow')
app.add_module(Input('start'))
app.add_module(Output('end'))
app.add_module(planner)
app.add_module(executor)

def join_steps(steps: List[str]):
    steps_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
    return {"steps": steps_str}

join_steps_box = CodeBox('join_steps', join_steps)
app.add_module(join_steps_box)
app.add_edge('start', 'planner')
app.add_edge('planner', 'join_steps')
app.add_edge('join_steps', 'executor')
app.add_edge('executor', 'end')
app.compile()

@register_opt_program_entry
def trial(input: dict):
    parser = argparse.ArgumentParser(description="script args")
    parser.add_argument("--s-arg", help="test")
    
    args = parser.parse_args()
    print(f"Script args: {args}")
    
    state = StatePool()
    state.init(input)
    app.invoke(state)
    print(f'Steps: {state.news("steps")}')
    print(f'Response: {state.news("response")}')
    return state.news('response')

@register_opt_score_fn
def score_fn(label, pred: str):
    return random.uniform(0.0, 1.0)

# from compiler.optimizer.evaluation.metric import ExactMatch
# metric = ExactMatch()

if __name__ == "__main__":
    print('Running directly')
