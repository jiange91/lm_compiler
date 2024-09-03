from compiler.utils import load_api_key, get_bill
from compiler.IR.program import Workflow, hint_possible_destinations, Context
from compiler.IR.base import Module, StatePool
from compiler.IR.modules import Retriever, Input, Output, Identity, CodeBox, Branch
from compiler.IR.rewriter.graph_inline import GraphInliner
from compiler.langchain_bridge.interface import LangChainLM
from pprint import pprint
import os
import json
import logging

logger = logging.getLogger(__name__)

load_api_key('secrets.toml')

from query_expansion import query_expansion_lm
from plot_coder import workspace_structure_codebox, initial_coder_lm, execute_and_log, collect_error_message, plot_debugger
from visual_refine import visual_refinement, img_encode_codebox, refine_plot_coder

def as_initial_coder():
    return {'current_role': 'initial_coder'}
def as_visual_refinement():
    return {'current_role': 'refine_coder'}

def prepare_plot_env(sample_id, workspace, data_path):
    new_directory_path = f'{workspace}/{sample_id}_{uuid.uuid4()}'
    if not os.path.exists(new_directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(new_directory_path, exist_ok=True)
        input_path = f'{data_path}/data/{sample_id}'
        if os.path.exists(input_path):
            #全部copy到f"Directory '{directory_path}'
            os.system(f'cp -r {input_path}/* {new_directory_path}')
    else:
        logger.error(f"Directory '{new_directory_path}' already exists.")
    return {'workspace': new_directory_path}

matplot_flow = Workflow('matplot')
matplot_flow.add_module(Input('start'))
matplot_flow.add_module(Output('end'))
matplot_flow.add_module(CodeBox('prepare plot env', prepare_plot_env))
matplot_flow.add_module(query_expansion_lm)
matplot_flow.add_module(workspace_structure_codebox)
matplot_flow.add_module(initial_coder_lm)
matplot_flow.add_module(execute_and_log)
matplot_flow.add_module(collect_error_message)
matplot_flow.add_module(plot_debugger)
matplot_flow.add_module(img_encode_codebox)
matplot_flow.add_module(visual_refinement)
matplot_flow.add_module(refine_plot_coder)
matplot_flow.add_module(CodeBox('as initial coder', as_initial_coder))
matplot_flow.add_module(CodeBox('as refine coder', as_visual_refinement))

matplot_flow.add_edge('start', 'prepare plot env')
matplot_flow.add_edge('prepare plot env', 'query expansion')
matplot_flow.add_edge('query expansion', 'workspace inspector')
matplot_flow.add_edge('workspace inspector', 'initial code generation')
matplot_flow.add_edge('initial code generation', 'as initial coder')
matplot_flow.add_edge('as initial coder', 'execute and log')
matplot_flow.add_edge('execute and log', 'collect error message')

matplot_flow.add_module(Identity('pass')) # when adding edge the compiler will check if module exists, add this to bypass the check


@hint_possible_destinations(['pass', 'plot debugger', 'end'])
def branch_on_error(ctx: Context, error_message, current_role):
    print(f"current role: {current_role}, error_message: {error_message}")
    if ctx.invoke_time >= 4:
        branch: Branch = ctx.calling_module
        branch.invoke_times = -1 # will add 1 after this function returns
        if current_role == 'initial_coder':
            return 'pass'
        else:
            return 'end'
    if error_message is not None:
        return 'plot debugger'
    else:
        branch: Branch = ctx.calling_module
        branch.invoke_times = -1 # will add 1 after this function returns
        if current_role == 'initial_coder':
            return 'pass'
        else:
            return 'end'

matplot_flow.add_branch('error handling', 'collect error message', branch_on_error)
matplot_flow.add_edge('plot debugger', 'execute and log')

@hint_possible_destinations(['img encode', 'end'])
def can_do_visual_refinement(ctx, workspace, plot_file_name):
    if os.path.exists(os.path.join(workspace, plot_file_name)):
        return 'img encode'
    else:
        return 'end'

matplot_flow.add_branch('if img exists', 'pass', can_do_visual_refinement)
matplot_flow.add_edge('img encode', 'visual refinement')
matplot_flow.add_edge('visual refinement', 'visual refine coder')
matplot_flow.add_edge('visual refine coder', 'as refine coder')
matplot_flow.add_edge('as refine coder', 'execute and log')

matplot_flow.compile()
matplot_flow.visualize('examples/matplot_agent/matplot_flow_viz')

lm = 'gpt-4o-mini'
openai_kwargs = {
    'temperature': 0,
}

query_expansion_lm.lm_config = {'model': lm, **openai_kwargs}
initial_coder_lm.lm_config = {'model': lm, **openai_kwargs}
plot_debugger.lm_config = {'model': lm, **openai_kwargs}
visual_refinement.lm_config = {'model': lm, **openai_kwargs}
refine_plot_coder.lm_config = {'model': lm, **openai_kwargs}

import uuid

def load_data():
    data_path = 'examples/matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    states = []
    for item in data:
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id'] 
        state = StatePool()
        directory_path = 'examples/matplot_agent/compiler_logs/runs'

        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
        state.init(
            {
                'query': novice_instruction,
                "workspace": directory_path,
                "plot_file_name": "plot.png",
                "sample_id": example_id,
                "data_path": data_path
            }
        )
        states.append(state)
    return states

bench_states = load_data()

from compiler.optimizer.tracer import OfflineBatchTracer
from evaluator import vision_score

def infinite_none():
    while True:
        yield None

log_dir = 'examples/matplot_agent/compiler_logs'

def get_original_trace():
    tracer = OfflineBatchTracer(
        workflow=matplot_flow,
        module_2_config='gpt-4o-2024-05-13',
        final_output_metric=vision_score,
    )
    trials = tracer.run(
        inputs=bench_states,
        labels=infinite_none(),
        log_dir=log_dir,
    )
    return trials

teacher_trials = get_original_trace()
# exit()

# ========================================
# Optimization
# ========================================

lm_options = [
    'gpt-4o-mini', # cheap
    'gpt-4o-2024-08-06', # medium
    'gpt-4o-2024-05-13', # expensive
]

from compiler.optimizer.model_selection_bo import LMSelectionBayesianOptimization
from compiler.optimizer.importance_eval import LMImportanceEvaluator
from compiler.optimizer.decompose import LMTaskDecompose

def eval_importance():
    evaluator = LMImportanceEvaluator(
        workflow=matplot_flow,
        models=['gpt-4o-mini', 'gpt-4o-2024-05-13'],
        base_model='gpt-4o-mini',
        final_output_metric=vision_score,
        trainset_input=[bench_states[1]],
        trainset_label=infinite_none(),
    )
    important_lms = evaluator.eval(
        log_dir=log_dir,
    )
    print(important_lms)
    return important_lms

important_lms = eval_importance()
# exit()

# Select model for each module
def select_models():
    selector = LMSelectionBayesianOptimization(
        workflow=matplot_flow,
        teachers='gpt-4o-2024-05-13',
        module_2_options=lm_options,
        final_output_metric=vision_score,
        trainset_input=[bench_states[1]],
        trainset_label=infinite_none(),
    )
    
    selector.optimize(
        n_trials=10,
        log_dir=log_dir,
        base_model='gpt-4o-mini',
        important_lms=important_lms,
    )

select_models()
exit()

# ========================================
# Sample run
# ========================================

def sample_run(example_id, query, workspace, plot_file_name):
    state = StatePool()
    state.init({
        'query': query,
        "workspace": workspace,
        "plot_file_name": plot_file_name
    })
    matplot_flow.reset()
    matplot_flow.pregel_run(state)
    logger.info(f"Example {example_id} finished.")

from tqdm import tqdm



def testing():
    states = load_data()
    
    for state in states:
        matplot_flow.reset()
        matplot_flow.pregel_run(state)
        print(vision_score(None, state))
