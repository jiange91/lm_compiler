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
import uuid

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
    return {'workspace': new_directory_path, "try_count": 0}



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
    # print(f"current role: {current_role}, error_message: {error_message}")
    print(f"current role: {current_role}")
    if ctx.invoke_time >= 4:
        branch: Branch = ctx.calling_module
        branch.invoke_times = -1 # will add 1 after this function returns
        if current_role == 'initial_coder':
            return 'pass'
        else:
            return 'end'
    if error_message != "":
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

lm = 'gpt-4o'
openai_kwargs = {
    'temperature': 0,
}

query_expansion_lm.lm_config = {'model': lm, **openai_kwargs}
initial_coder_lm.lm_config = {'model': lm, **openai_kwargs}
plot_debugger.lm_config = {'model': lm, **openai_kwargs}
visual_refinement.lm_config = {'model': lm, **openai_kwargs}
refine_plot_coder.lm_config = {'model': lm, **openai_kwargs}


from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.params.meta_programming.mp import MetaPrompting

# peng = MetaPrompting()
# peng.apply(query_expansion_lm)
# # peng.apply(initial_coder_lm)
# peng.apply(visual_refinement)

# ========================================
# Sample run
# ========================================

def sample_run(id):
    data_path = 'examples/matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    for item in data:
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id']
        if example_id == id:
            state = StatePool()
            directory_path = 'examples/matplot_agent/runs'

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
            matplot_flow.reset()
            matplot_flow.pregel_run(state)
            logger.info(f"Visual refinement: {state.news('visual_refinement')}")
    
# sample_run(95)
# exit()

# ========================================
# Evaluation
# ========================================
from tqdm import tqdm
from evaluator import VisionScore
from compiler.optimizer.evaluation.evaluator import Evaluator

def load_data(result_dir='examples/matplot_agent/sample_runs'):
    matplot_flow.reset()
    data_path = 'examples/matplot_agent/benchmark_data'
    # open the json file 
    data = json.load(open(f'{data_path}/benchmark_instructions_minor.json'))
    
    states = []
    for item in data:
        novice_instruction = item['simple_instruction']
        expert_instruction = item['expert_instruction']
        example_id = item['id'] 
        state = StatePool()
        directory_path = result_dir

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

def testing():
    states = load_data()
    eval_set = [(state, None) for state in states]
    evaluator = Evaluator(
        metric=VisionScore(),
        eval_set=eval_set,
        num_thread=3,
    )
    states, avg_score, avg_price = evaluator(workflow=matplot_flow)
    print(f"Average score: {avg_score}, Average price: {avg_price}")

# testing()
# exit()

# ========================================
# Importance evaluation
# ========================================
from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common

def importance_eval():
    importance_evaluator = LMImportanceEvaluator(
        workflow=matplot_flow,
        options=model_selection.model_option_factory(['gpt-4o-mini', 'gpt-4o'])
    )
    states = load_data()
    eval_set = [(state, None) for state in states]
    evaluator = Evaluator(
        metric=VisionScore(),
        eval_set=eval_set,
        num_thread=3,
    )
    important_lms = importance_evaluator.eval(evaluator, 'examples/matplot_agent/optimizer_logs')
    return important_lms

important_lms = importance_eval()
# exit()

# ========================================
# Optimization
# ========================================
from compiler.optimizer.layered_optimizer import GeneralLLMLayer
from compiler.optimizer.driver import LayerDriver

def opt():
    lm_options = [
        'gpt-4o-mini',
        'gpt-4o',
    ]
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [common.IdentityOption(), ZeroShotCoT(), PlanBefore()]
    )

    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    quality_cons = lambda x: x >= 0.95
    reasoning_layer = GeneralLLMLayer(
        name='prompt_reasoning',
        params=[reasoning_param],
        quality_constraint=quality_cons,
        opt_direction='maximize',
    )
    
    model_selection_layer = GeneralLLMLayer(
        name='model_selection',
        params=[model_param],
        quality_constraint=quality_cons,
        opt_direction='maximize',
    )
    
    states = load_data(result_dir='examples/matplot_agent/layer_optimizer_runs')
    eval_set = [(state, None) for state in states]
    evaluator = Evaluator(
        metric=VisionScore(),
        eval_set=eval_set,
        num_thread=3,
    )


    opt_driver = LayerDriver(
        [reasoning_layer, model_selection_layer],
        budgets=[2, 1],
        evaluator=evaluator,
        opt_direction='maximize',
    )

    opt_driver.fire(matplot_flow, 'examples/matplot_agent/layer_optimizer_logs')

opt()