from compiler.utils import load_api_key
from compiler.IR.program import Workflow, hint_possible_destinations, Context
from compiler.IR.base import Module, StatePool
from compiler.IR.modules import Retriever, Input, Output
from compiler.langchain_bridge.interface import LangChainLM
from pprint import pprint

load_api_key('secrets.toml')

from direct_hyde import direct_hyde_semantic
from retrieve import retrieve_kernel
from doc_filter import sub_question_mapper, doc_filter_lm, doc_filter_post_process
from generator import sub_answer_mapper, generator_lm
from subanswer_compose import answer_compose_lm, knowledge_preprocess
from kalmv import kalmv_module
# from query_rewriter import query_rewriter_kernel

direct_hyde_module = LangChainLM('direct_hyde', direct_hyde_semantic)
retrieve_module = Retriever('retrieve', retrieve_kernel)
# answer_compose_module = LangChainLM('answer_compose', answer_compose_kernel)
# kalmv_module = LangChainLM('kalmv', kalmv_kernel)

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
    
rag_workflow.add_branch('kalmv', final_router)
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

    rag_workflow.pregel_run(state)
    print(state.news('answer'))
    rag_workflow.log_token_usage('examples/langraph_rag_src/token_usage.json')
    rag_workflow.log_module_time('examples/langraph_rag_src/module_time.json')

# sample_run()
# exit()

# --------------------------------------------
# Optimization
# --------------------------------------------

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

from self_eval import evaluate_rag_answer_compatible
from compiler.optimizer.model_selection_bo import LMSelectionBayesianOptimization
from compiler.optimizer.importance_eval import LMImportanceEvaluator
from compiler.optimizer.decompose import LMTaskDecompose

state = StatePool()
state.init({'question': "What are the types of agent memory?"})
def infinite_none():
    while True:
        yield None


# Decompose LM Modules
def task_disambiguous():
    decomposer = LMTaskDecompose(
        workflow=rag_workflow,
    )
    decomposer.decompose(4)

task_disambiguous()
exit()

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
        log_dir='examples/langraph_rag_src/compile_log',
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
        n_trials=10,
        log_dir='examples/langraph_rag_src/compile_log',
        base_model='gpt-4o-mini',
        important_lms=important_lms,
    )

select_models()