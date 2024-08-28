from compiler.utils import load_api_key
from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import Retriever
from compiler.langchain_bridge.interface import LangChainLM
from pprint import pprint


load_api_key('secrets.toml')

from query_decompose import decompose_kernel
from hyde import hyde_kernel
from direct_hyde import direct_hyde_kernel
from adaptive_routing import router_kernel
# from doc_retrive_filter import retrieve_filter_kernel
from retrieve import retrieve_kernel
from doc_filter import doc_filter_kernel
from generator import generator_kernel
from subanswer_compose import answer_compose_kernel

from hulc_grader import hulc_grader_kernel
from answer_grader import answer_grader_kernel
from kalmv import kalmv_kernel

from compiler.optimizer.model_selection_bo import HOLMSelection
from compiler.utils import compare_two_answer
# from query_rewriter import query_rewriter_kernel

decompose_module = LangChainLM('decompose', decompose_kernel)
hyde_module = LangChainLM('hyde', hyde_kernel)
direct_hyde_module = LangChainLM('direct_hyde', direct_hyde_kernel)
# route_query_module = LangChainLM('route_query', router_kernel)
# retrieve_filter_module = LangChainLM('retrieve_filter', retrieve_filter_kernel)
retrieve_module = Retriever('retrieve', retrieve_kernel)
doc_filter_module = LangChainLM('doc_filter', doc_filter_kernel)
generator_module = LangChainLM('knowledge curation', generator_kernel)
answer_compose_module = LangChainLM('answer_compose', answer_compose_kernel)
hulc_grader_module = LangChainLM('hulc_grader', hulc_grader_kernel)
answer_grader_module = LangChainLM('answer_grader', answer_grader_kernel)
# query_rewriter_module = LangChainLM('query_rewriter', query_rewriter_kernel)
kalmv_module = LangChainLM('kalmv', kalmv_kernel)


rag_workflow = Workflow()
rag_workflow.add_module(decompose_module)
rag_workflow.add_module(hyde_module)
# rag_workflow.add_module(direct_hyde_module)
# rag_workflow.add_module(route_query_module)
# rag_workflow.add_module(retrieve_filter_module)
rag_workflow.add_module(retrieve_module)
rag_workflow.add_module(doc_filter_module)
rag_workflow.add_module(generator_module)
rag_workflow.add_module(answer_compose_module)
# rag_workflow.add_module(hulc_grader_module)
# rag_workflow.add_module(answer_grader_module)
# rag_workflow.add_module(query_rewriter_module)
# rag_workflow.add_module(kalmv_module)

rag_workflow.add_edge(decompose_module, hyde_module)
rag_workflow.add_edge(hyde_module, retrieve_module)
# rag_workflow.add_edge(direct_hyde_module, retrieve_module)
rag_workflow.add_edge(retrieve_module, doc_filter_module)
rag_workflow.add_edge(doc_filter_module, generator_module)
rag_workflow.add_edge(generator_module, answer_compose_module)
# rag_workflow.add_edge(answer_compose_module, kalmv_module)
# rag_workflow.add_edge(answer_compose_module, hulc_grader_module)
# rag_workflow.add_edge(hulc_grader_module, answer_grader_module)
# rag_workflow.add_edge(generator_module, answer_grader_module)
# rag_workflow.add_edge(answer_grader_module, query_rewriter_module)
# rag_workflow.set_exit_point(query_rewriter_module, 'question')

rag_workflow.set_exit_point(answer_compose_module, 'answer')

openai_kwargs = {
    'temperature': 0,
}

sample_lm = 'gpt-4o-mini'
decompose_module.lm_config = {'model': 'gpt-4o', **openai_kwargs}
hyde_module.lm_config = {'model': sample_lm, **openai_kwargs}
direct_hyde_module.lm_config = {'model': sample_lm, **openai_kwargs}
# route_query_module.lm_config = {'model': sample_lm, **openai_kwargs}
# retrieve_filter_module.lm_config = {'model': sample_lm, **openai_kwargs}
retrieve_module.lm_config = {'model': sample_lm, **openai_kwargs}
doc_filter_module.lm_config = {'model': sample_lm, **openai_kwargs}
generator_module.lm_config = {'model': sample_lm, **openai_kwargs}
answer_compose_module.lm_config = {'model': sample_lm, **openai_kwargs}
kalmv_module.lm_config = {'model': sample_lm, **openai_kwargs}
hulc_grader_module.lm_config = {'model': sample_lm, **openai_kwargs}
# query_rewriter_module.lm_config = {'model': sample_lm, **openai_kwargs}

state = StatePool()
state.publish({'question': "What are the types of agent memory?"})
# state.publish({'question': "What's the financial performance of Nvidia and Apple over the past three years? Which company has a higher market capitalization?"})

# rag_workflow.run(state=state)
# print(state.state)
# rag_workflow.log_token_usage('token_usage.json')
# rag_workflow.log_module_time('module_time.json')
# exit()

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

from self_eval import *

ms_boot = HOLMSelection(
    workflow=rag_workflow,
    teachers='gpt-4o',
    module_2_options=lm_options,
    final_output_metric=evaluate_cosine_sim,
    trainset_input=[state],
    trainset_label=None,
)

prior_trials = [
    {
        "decompose": "gpt-4o",
        "hyde": "gpt-4o-mini",
        "doc_filter": "gpt-4o-mini",
        "knowledge curation": "gpt-4o-mini",
        "answer_compose": "gpt-4o",
    },
    {
        "decompose": "gpt-4o-mini",
        "hyde": "gpt-4o-mini",
        "doc_filter": "gpt-4o-mini",
        "knowledge curation": "gpt-4o-mini",
        "answer_compose": "gpt-4o",
    },
    {
        "decompose": "gpt-4o",
        "hyde": "gpt-4o-mini",
        "doc_filter": "gpt-4o-mini",
        "knowledge curation": "gpt-4o-mini",
        "answer_compose": "gpt-4o-mini",
    },
    {
        "decompose": "gpt-4o-mini",
        "hyde": "gpt-4o-mini",
        "doc_filter": "gpt-4o-mini",
        "knowledge curation": "gpt-4o-mini",
        "answer_compose": "gpt-4o-mini",
    }
]

solutions = ms_boot.compile(
    log_dir='rag_model_selection_bo_prior_trials',
    gap=0.0,
    prior_trials=prior_trials,
)