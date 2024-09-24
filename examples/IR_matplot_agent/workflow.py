import argparse
import json
import re
import uuid
import dspy

from tqdm import tqdm
from agents.query_expansion_agent.agent import QueryExpansionAgent, query_expansion_agent
from agents.plot_agent.agent import PlotAgent, PlotAgentModule
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
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn

load_api_key('secrets.toml')

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='gpt-4o-mini')
parser.add_argument('--visual_refine', type=bool, default=True)
args = parser.parse_args()

@register_opt_program_entry
def mainworkflow(input: dict):
    query, directory_path, example_id, input_path = input['query'], input['directory_path'], input['example_id'], input['input_path']
    # Prepare workspace
    workspace = f'{directory_path}/{example_id}_{uuid.uuid4().hex}'
    if not os.path.exists(workspace):
        # If it doesn't exist, create the directory
        os.makedirs(workspace, exist_ok=True)
        if os.path.exists(input_path):
            os.system(f'cp -r {input_path}/* {workspace}')
    else:
        logging.info(f"Directory '{workspace}' already exists.")
        
    # Query expanding
    logging.info('=========Query Expansion AGENT=========')
    config = {'workspace': workspace}
    expanded_simple_instruction = query_expansion_agent.invoke({'query': query}).content
    # logging.info('=========Expanded Simple Instruction=========')
    # logging.info(expanded_simple_instruction)
    logging.info('=========Plotting=========')

    # GPT-4 Plot Agent
    # Initial plotting
    action_agent = PlotAgentModule()
    logging.info('=========Novice 4 Plotting=========')
    novice_log, novice_code = action_agent.run(
        query=query,
        expanded_query=expanded_simple_instruction,
        query_type='initial',
        file_name='novice.png',
        workspace=workspace,
    )
    logging.info(novice_log)
    # logging.info('=========Original Code=========')
    # logging.info(novice_code)

    # Visual refinement
    if os.path.exists(f'{workspace}/novice.png'):
        print('Use original code for visual feedback')
        visual_refine_agent = VisualRefineAgent('novice.png', config, '', query)
        visual_feedback = visual_refine_agent.run('gpt-4o-mini', 'novice', 'novice_final.png')
        logging.info('=========Visual Feedback=========')
        logging.info(visual_feedback)
        final_instruction = '' + '\n\n' + visual_feedback
        
        novice_log, novice_code = action_agent.run(
            query=query,
            expanded_query=final_instruction,
            query_type='refinement',
            file_name='novice_final.png',
            workspace=workspace,
        )
        logging.info(novice_log)
    return {
        "img_path": f"{workspace}/novice_final.png",
        "rollback": f"{workspace}/novice.png",
    }

from evaluator import gpt_4v_evaluate

@register_opt_score_fn
def matplot_eval(gold, pred) -> float:
    return gpt_4v_evaluate(gold['ground_truth'], pred['img_path'], pred['rollback'])

if __name__ == "__main__":
    print("-- Running main workflow --")
    data_path = 'examples/IR_matplot_agent/benchmark_data'
    
    # open the json file 
    data = json.load(open(f'{data_path}/82.json'))
    
    for item in tqdm(data):
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id']
        directory_path = "examples/IR_matplot_agent/sample_runs_direct"

        # Check if the directory already exists
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
        mainworkflow({
            'query': novice_instruction,
            'directory_path': directory_path,
            'example_id': example_id,
            'input_path': f'{data_path}/data/{example_id}',
        })