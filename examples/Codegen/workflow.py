import dotenv
from langchain_openai import ChatOpenAI

# Load the environment variables
dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


from langchain_core.prompts import ChatPromptTemplate
#================= Complete Code Agent =================
cc_system_prompt = """
Given an incomplete python function, complete the function body according to the provided docstring. Only the function body should be returned, and wrapped in <result> and </result> tags.
"""
    
cc_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", cc_system_prompt),
        ("human", "incomplete function:\n\n{incomplete_function}"),
    ]
)

cc_agent = cc_agent_prompt | model

#================= Refine Code Agent =================
rc_system_prompt = """
Given an incomplete python function and the function body generated by another agent, improve the code completion. Only the function body should be returned, and wrapped in <result> and </result> tags.
"""

rc_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rc_system_prompt),
        ("human", "incomplete function:\n\n{incomplete_function}\n\ncompleted code:\n\n{completed_code}"),
    ]
)

rc_agent = rc_agent_prompt | model

#================= CodeGen Workflow =================

import cognify

@cognify.register_workflow
def codegen_workflow(problem):
    incomplete_function = problem['prompt']
    complete_code = cc_agent.invoke({"incomplete_function": incomplete_function}).content
    finalized_code = rc_agent.invoke({"incomplete_function": incomplete_function, "completed_code": complete_code}).content
    return {"finalized_code": finalized_code}


if __name__ == '__main__':
    incomplete_function = '''
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
    """
'''
    print(codegen_workflow({'prompt': incomplete_function})['finalized_code'])