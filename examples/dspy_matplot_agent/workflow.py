import argparse
import json
import re
import uuid
import dspy

from tqdm import tqdm
from agents.query_expansion_agent import QueryExpansionAgent, QueryExpansionModule
from agents.plot_agent.agent import PlotAgent, PlotAgentModule, Debugger, PlotRefiner, PlotCoder
from agents.visual_refine_agent import VisualRefineAgent
import logging
import os
import shutil
import glob
import sys
from agents.utils import is_run_code_success, run_code, get_code
from agents.dspy_common import OpenAIModel
from agents.config.openai import openai_kwargs
from compiler.utils import load_api_key, get_bill

load_api_key('secrets.toml')

from plot_module import MatPlotModule

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='gpt-4o-mini')
parser.add_argument('--visual_refine', type=bool, default=True)
args = parser.parse_args()

def mainworkflow(expert_instruction, simple_instruction, workspace, max_try=3):
    plot_module = MatPlotModule(args.model_type)
    plot_module(simple_instruction, workspace)


def check_refined_code_executable(refined_code, model_type, query_type, workspace):
    file_name = f'code_action_{model_type}_{query_type}_refined.py'
    with open(os.path.join(workspace, file_name), 'w') as f1:
        f1.write(refined_code)
    log = run_code(workspace, file_name)

    return is_run_code_success(log)


if __name__ == "__main__":
    data_path = 'examples/dspy_matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    for item in tqdm(data):
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id']
        directory_path = f'examples/dspy_matplot_agent/sample_runs/{example_id}_{uuid.uuid4().hex}'

        # Check if the directory already exists
        if not os.path.exists(directory_path):
            # If it doesn't exist, create the directory
            os.makedirs(directory_path, exist_ok=True)
            print(f"Directory '{directory_path}' created successfully.")
            input_path = f'{data_path}/data/{example_id}'
            if os.path.exists(input_path):
                #全部copy到f"Directory '{directory_path}'
                os.system(f'cp -r {input_path}/* {directory_path}')
        else:
            print(f"Directory '{directory_path}' already exists.")
            continue
        mainworkflow(expert_instruction, novice_instruction, workspace=directory_path)
