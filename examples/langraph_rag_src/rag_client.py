from compiler.utils import load_api_key, get_bill
from compiler.IR.program import Workflow, hint_possible_destinations, Context
from compiler.IR.base import Module, StatePool
from compiler.IR.modules import Retriever, Input, Output
from compiler.IR.rewriter.graph_inline import GraphInliner
from compiler.langchain_bridge.interface import LangChainLM
from pprint import pprint
import json

load_api_key('secrets.toml')

from direct_hyde import direct_hyde_module
from retrieve import retrieve_kernel
from doc_filter import sub_question_mapper, doc_filter_lm, doc_filter_post_process
from generator import sub_answer_mapper, generator_lm
from subanswer_compose import answer_compose_lm, knowledge_preprocess
from kalmv import kalmv_module
# from query_rewriter import query_rewriter_kernel

retrieve_module = Retriever('retrieve', retrieve_kernel)

rag_workflow = Workflow('rag')
rag_workflow.add_module(Input('query'))
rag_workflow.add_module(direct_hyde_module)
rag_workflow.add_module(retrieve_module)
rag_workflow.add_module(sub_question_mapper)
rag_workflow.add_module(doc_filter_post_process)
rag_workflow.add_module(sub_answer_mapper)
rag_workflow.add_module(knowledge_preprocess)
rag_workflow.add_module(answer_compose_lm)
rag_workflow.add_module(kalmv_module)
rag_workflow.add_module(Output('answer'))

rag_workflow.add_edge('query', 'direct_hyde')
rag_workflow.add_edge('direct_hyde', 'retrieve')
rag_workflow.add_edge('retrieve', 'sub_question_mapper')
rag_workflow.add_edge('sub_question_mapper', 'doc_filter_post_process')
rag_workflow.add_edge('doc_filter_post_process', 'knowledge_curation')
rag_workflow.add_edge('knowledge_curation', 'knowledge_preprocess')
rag_workflow.add_edge('knowledge_preprocess', 'answer_compose')
rag_workflow.add_edge('answer_compose', 'kalmv')

@hint_possible_destinations(['direct_hyde', 'answer_compose', 'answer'])
def final_router(ctx: Context, decision: str):
    if ctx.invoke_time >= 3:
        return 'answer'
    if decision == 'ae' or decision == 're':
        return 'direct_hyde'
    if decision == 'ge':
        return 'answer_compose'
    return 'answer'
    
rag_workflow.add_branch('answer_verification', 'kalmv', final_router)
rag_workflow.compile()


openai_kwargs = {
    'temperature': 0,
}

sample_lm = 'gpt-4o-mini'
direct_hyde_module.lm_config = {'model': sample_lm, **openai_kwargs}
doc_filter_lm.lm_config = {'model': sample_lm, **openai_kwargs}
generator_lm.lm_config = {'model': sample_lm, **openai_kwargs}
answer_compose_lm.lm_config = {'model': sample_lm, **openai_kwargs}
kalmv_module.lm_config = {'model': sample_lm, **openai_kwargs}

# --------------------------------------------
# Sample run
# --------------------------------------------
def sample_run():
    state = StatePool()
    state.init({'question': "What are the types of agent memory?"})
    # state.publish({'question': "What's the financial performance of Nvidia and Apple over the past three years? Which company has a higher market capitalization?"})

    rag_workflow.reset()
    rag_workflow.pregel_run(state)
    
    print(state.news('answer'))
    rag_workflow.log_token_usage('examples/langraph_rag_src/token_usage.json')
    rag_workflow.log_module_time('examples/langraph_rag_src/module_time.json')

sample_run()
# rag_workflow.log_token_usage('examples/langraph_rag_src/token_usage.json')
# print(get_bill(rag_workflow.token_usage_buffer))


# rag_workflow.visualize('examples/langraph_rag_src/rag_workflow_viz')

exit()

log_dir = 'examples/langraph_rag_src/compile_log_1'
# --------------------------------------------
# Get original workflow trace
# --------------------------------------------
state = StatePool()
state.init({'question': "What are the types of agent memory?"})

from self_eval import evaluate_rag_answer_compatible
from compiler.optimizer.tracer import OfflineBatchTracer

def infinite_none():
    while True:
        yield None

def get_original_trace():
    tracer = OfflineBatchTracer(
        workflow=rag_workflow,
        module_2_config='gpt-4o',
        final_output_metric=evaluate_rag_answer_compatible,
    )
    trials = tracer.run(
        inputs=[state],
        labels=infinite_none(),
        field_in_interest=['answer'],
        log_dir=log_dir,
    )
    return trials

original_trials = get_original_trace()
# exit()
# --------------------------------------------
# Optimization
# --------------------------------------------

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

from compiler.optimizer.model_selection_bo import LMSelectionBayesianOptimization
from compiler.optimizer.importance_eval import LMImportanceEvaluator
from compiler.optimizer.decompose import LMTaskDecompose


# Decompose LM Modules
def task_disambiguous():
    decomposer = LMTaskDecompose(
        workflow=rag_workflow,
    )
    decomposer.decompose(
        log_dir=log_dir,
        threshold=3,
    )

task_disambiguous()

def inline_graph():
    inliner = GraphInliner(
        workflow=rag_workflow,
    )
    inliner.flatten()
    rag_workflow.compile()

inline_graph()

from compiler.IR.utils import get_function_kwargs, simple_cycles
# sample_run()
cycles = sorted(simple_cycles(rag_workflow.edges), key=lambda x: len(x), reverse=True)


# Find important LMs

def eval_importance():
    evaluator = LMImportanceEvaluator(
        workflow=rag_workflow,
        models=lm_options,
        base_model='gpt-4o-mini',
        final_output_metric=evaluate_rag_answer_compatible,
        trainset_input=[state],
        trainset_label=infinite_none(),
    )
    important_lms = evaluator.eval(
        log_dir=log_dir,
    )
    print(important_lms)
    return important_lms

important_lms = eval_importance()
# Select model for each module
def select_models():
    selector = LMSelectionBayesianOptimization(
        workflow=rag_workflow,
        teachers='gpt-4o',
        module_2_options=lm_options,
        final_output_metric=evaluate_rag_answer_compatible,
        trainset_input=[state],
        trainset_label=infinite_none(),
    )
    
    selector.optimize(
        n_trials=4,
        log_dir=log_dir,
        base_model='gpt-4o-mini',
        important_lms=important_lms,
        fields_in_interest=['answer'],
    )

select_models()