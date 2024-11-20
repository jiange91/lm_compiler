import dotenv
from cognify._logging import _configure_logger

dotenv.load_dotenv()
_configure_logger("INFO")

from typing import List

from cognify.llm.model import LMConfig

lm_config = LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

from cognify.llm.model import Model, Input, OutputLabel
#================= Complete Code Agent =================
cc_system_prompt = """
Given an incomplete python function, complete the function body according to the provided docstring.
"""
    
cognify_cc_agent = Model(
    agent_name="cc_agent",
    system_prompt=cc_system_prompt,
    input_variables=[
        Input(name="incomplete_function"),
    ],
    output=OutputLabel(
        name="completed_code",
        custom_output_format_instructions="Only the function body should be returned, and wrapped in <result> and </result> tags."
    ),
    lm_config=lm_config,
)

from cognify.frontends.langchain.connector import as_runnable
cc_agent = as_runnable(cognify_cc_agent)

#================= Refine Code Agent =================
rc_system_prompt = """
Given an incomplete python function and the function body generated by another agent, improve the code completion. 
"""

cognify_rc_agent = Model(
    agent_name="rc_agent",
    system_prompt=rc_system_prompt,
    input_variables=[
        Input(name="incomplete_function"),
        Input(name="completed_code"),
    ],
    output=OutputLabel(
        name="finalized_code",
        custom_output_format_instructions="Only the function body should be returned, and wrapped in <result> and </result> tags.",
    ),
    lm_config=lm_config,
)

rc_agent = as_runnable(cognify_rc_agent)

#================= CodeGen Workflow =================

def cc_agent_routine(state):
    func = state["incomplete_function"]
    return {"completed_code": cc_agent.invoke({"incomplete_function": func}).content}

def rc_agent_routine(state):
    func = state["incomplete_function"]
    completed_code = state["completed_code"]
    return {"finalized_code": rc_agent.invoke({"incomplete_function": func, "completed_code": completed_code}).content}

from langgraph.graph import END, START, StateGraph, MessagesState
from typing import Dict, TypedDict

class State(TypedDict):
    incomplete_function: str
    completed_code: str
    finalized_code: str
    
workflow = StateGraph(State)
workflow.add_node("complete_code", cc_agent_routine)
workflow.add_node("refine_code", rc_agent_routine)
workflow.add_edge(START, "complete_code")
workflow.add_edge("complete_code", "refine_code")
workflow.add_edge("refine_code", END)

app = workflow.compile()

from cognify.optimizer.registry import register_opt_program_entry

@register_opt_program_entry
def do_code_gen(input):
    state = app.invoke(
        {"incomplete_function": input}
    )
    return state['finalized_code']


if __name__ == '__main__':
    incomplete_function = \
'''
def correct_bracketing(brackets: str):
    """ brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
'''
    
    result = app.invoke({"incomplete_function": incomplete_function})
    
    print(result['incomplete_function'])
    print(result['finalized_code'])
    