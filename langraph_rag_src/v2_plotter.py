from compiler.utils import load_api_key
from compiler.IR.program import Workflow, Module, StatePool
from compiler.langchain_bridge.interface import LangChainLM

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

# from hulc_grader import hulc_grader_kernel
from answer_grader import answer_grader_kernel
from kalmv import kalmv_kernel
# from query_rewriter import query_rewriter_kernel

start_module = LangChainLM('start', decompose_kernel)
end_module = LangChainLM('end', decompose_kernel)
decompose_module = LangChainLM('decompose', decompose_kernel)
hyde_module = LangChainLM('hyde', hyde_kernel)
direct_hyde_module = LangChainLM('direct_hyde', direct_hyde_kernel)
retrieve_module = LangChainLM('retrieve', retrieve_kernel)
doc_filter_module = LangChainLM('doc_filter', doc_filter_kernel)
keyword_semantic_match = LangChainLM('keyword_semantic_match', doc_filter_kernel)
context_match = LangChainLM('contextual relevance', doc_filter_kernel)
generator_module = LangChainLM('knowledge curation', generator_kernel)
answer_compose_module = LangChainLM('answer_compose', answer_compose_kernel)
answer_grader_module = LangChainLM('answer_grader', answer_grader_kernel)
knowledge_eval_module = LangChainLM('knowledge relevant ?', kalmv_kernel)
answer_grounded_module = LangChainLM('answer grounded ?', answer_grader_kernel)
answer_eval_module = LangChainLM('question answered ?', answer_grader_kernel)
kalmv_module = LangChainLM('kalmv', kalmv_kernel)



rag_workflow = Workflow()
rag_workflow.add_module(start_module)
rag_workflow.add_module(decompose_module)
rag_workflow.add_module(hyde_module)
rag_workflow.add_module(retrieve_module)
rag_workflow.add_module(keyword_semantic_match)
rag_workflow.add_module(context_match)
rag_workflow.add_module(generator_module)
rag_workflow.add_module(answer_compose_module)
rag_workflow.add_module(knowledge_eval_module)
rag_workflow.add_module(answer_eval_module)
rag_workflow.add_module(end_module)

rag_workflow.add_edge(start_module, decompose_module)
rag_workflow.add_edge(decompose_module, hyde_module)
rag_workflow.add_edge(hyde_module, retrieve_module)

# rag_workflow.add_edge(start_module, direct_hyde_module)
# rag_workflow.add_edge(direct_hyde_module, retrieve_module)

# rag_workflow.add_edge(retrieve_module, doc_filter_module)
# rag_workflow.add_edge(doc_filter_module, generator_module)
rag_workflow.add_edge(retrieve_module, keyword_semantic_match)
rag_workflow.add_edge(retrieve_module, context_match)
rag_workflow.add_edge(keyword_semantic_match, generator_module)
rag_workflow.add_edge(context_match, generator_module)

rag_workflow.add_edge(generator_module, answer_compose_module)

rag_workflow.add_edge(answer_compose_module, knowledge_eval_module)
rag_workflow.add_edge(knowledge_eval_module, decompose_module)
rag_workflow.add_edge(knowledge_eval_module, answer_grounded_module)
rag_workflow.add_edge(answer_grounded_module, answer_compose_module)
rag_workflow.add_edge(answer_grounded_module, answer_eval_module)
rag_workflow.add_edge(answer_eval_module, decompose_module)
rag_workflow.add_edge(answer_eval_module, end_module)


rag_workflow.set_exit_point(retrieve_module, 'decision')

openai_kwargs = {
    'temperature': 0,
}

sample_lm = 'gpt-4o-mini'
# decompose_module.lm_config = {'model': sample_lm, **openai_kwargs}
# hyde_module.lm_config = {'model': sample_lm, **openai_kwargs}
direct_hyde_module.lm_config = {'model': sample_lm, **openai_kwargs}
# route_query_module.lm_config = {'model': sample_lm, **openai_kwargs}
# retrieve_filter_module.lm_config = {'model': sample_lm, **openai_kwargs}
retrieve_module.lm_config = {'model': sample_lm, **openai_kwargs}
doc_filter_module.lm_config = {'model': sample_lm, **openai_kwargs}
generator_module.lm_config = {'model': sample_lm, **openai_kwargs}
answer_compose_module.lm_config = {'model': sample_lm, **openai_kwargs}
# kalmv_module.lm_config = {'model': sample_lm, **openai_kwargs}
# hulc_grader_module.lm_config = {'model': sample_lm, **openai_kwargs}
# query_rewriter_module.lm_config = {'model': sample_lm, **openai_kwargs}

state = StatePool()
state.publish({'question': "What are the types of agent memory?"})
# state.publish({'question': "What's the financial performance of Nvidia and Apple over the past three years? Which company has a higher market capitalization?"})

# rag_workflow.run(state=state)
# print(state.state)
rag_workflow.visualize('rag_workflow')