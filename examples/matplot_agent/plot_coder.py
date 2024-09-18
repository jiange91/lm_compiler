from typing import Literal
import concurrent.futures
from functools import wraps
import re
import os

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.modules import Map, CodeBox

from utils import print_filesys_struture, run_code, is_run_code_success, get_error_message, get_code

@wraps(print_filesys_struture)
def print_filesys_structure_wrapper(**kwargs):
    return {'workspace_structure': print_filesys_struture(**kwargs)}

workspace_structure_codebox = CodeBox('workspace inspector', print_filesys_structure_wrapper)

INITIAL_SYSTEM_PROMPT = '''
You are a cutting-edge super capable code generation LLM. You will be given a natural language query, generate a runnable python code to satisfy all the requirements in the query. You can use any python library you want. 

If the query requires data manipulation from a csv file, process the data from the csv file and draw the plot in one piece of code.

In your code, when you complete a plot, remember to save it to a png file with given the 'plot_file_name'.
'''

class PlottingCode(BaseModel):
    """Python code for plotting"""
    code: str = Field(description="Python code")

initial_coder_semantic = LangChainSemantic(
    system_prompt=INITIAL_SYSTEM_PROMPT,
    inputs=['expanded_query', 'plot_file_name'],
    output_format="code",
    output_format_instructions="Please only return the python code. Wrap it with ```python and ``` to format it properly.",
)

initial_coder_lm = LangChainLM('initial code generation', initial_coder_semantic)

def get_run_log(code, workspace, current_role, try_count):
    clean_code = get_code(code)
    file_name = f'code_action_{current_role}_{try_count}.py'
    with open(os.path.join(workspace, file_name), 'w+') as f:
        f.write(clean_code)
        
    log = run_code(workspace, file_name)
    return {'log': log, 'try_count': try_count + 1}

execute_and_log = CodeBox('execute and log', get_run_log)

def get_err_message(workspace, log, plot_file_name):
    if is_run_code_success(log):
        if print_filesys_struture(workspace).find(plot_file_name) == -1:
            log = log + '\n' + 'No plot generated.'
            error_message = f'The expected file is not generated. When you complete a plot, remember to save it to a png file. The file name should be """{plot_file_name}"""'
        else:
            error_message = ""
    else:
        error_message = get_error_message(log)
    return {'error_message': error_message}

collect_error_message = CodeBox('collect error message', get_err_message)

DEBUG_SYSTEM_PROMPT = """
You are a cutting-edge super capable code debugger LLM. You will be given a user query about data visualization, a piece of existing python code for completing the task and an error message associate with this code. 

Your task is to fix the error. You can use any python library you want. The code should be executable and can at least generate a plot without any error.
"""

plot_debugger_semantic = LangChainSemantic(
    system_prompt=DEBUG_SYSTEM_PROMPT,
    inputs=['query', 'code', 'error_message'],
    output_format="code",
    output_format_instructions="Please only return the python code. Wrap it with ```python and ``` to format it properly.",
)

plot_debugger = LangChainLM('plot debugger', plot_debugger_semantic)