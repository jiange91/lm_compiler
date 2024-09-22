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
from planner import runnable_planner_anno as planner
from executor import runnable_exec_anno as executor
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
    
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("executor", execute_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "executor")

workflow.add_edge("executor", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

@register_opt_program_entry
def trial(input):
    parser = argparse.ArgumentParser(description="script args")
    parser.add_argument("--s-arg", help="test")
    
    args = parser.parse_args()
    print(f"Script args: {args}")
    

    result = app.invoke(input)
    print(f'Steps: {result["steps"]}')
    print(f'Response: {result["response"]}')
    return result['response']

@register_opt_score_fn
def score_fn(label, pred):
    return random.uniform(0.0, 1.0)

if __name__ == "__main__":
    print('Running directly')
