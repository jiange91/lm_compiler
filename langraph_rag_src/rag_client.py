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

from compiler.optimizer.bootstrap import BootStrapLMSelection
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

rag_workflow.run(state=state)
print(state.state)
rag_workflow.log_token_usage('token_usage.json')
rag_workflow.log_module_time('module_time.json')
exit()

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

def query_decompose_compare(gold, pred):
    new_gold, new_pred = {}, {}
    new_gold['sub_questions'] = ", ".join(gold['sub_questions'])
    new_pred['sub_questions'] = ", ".join(pred['sub_questions'])
    return compare_two_answer(new_gold, new_pred)

def hyde_retrieve_compare(gold, pred):
    def doc_concate(docs):
        return "\n\n".join(doc for doc in docs)
    score = 0
    
    for gold_docs, pred_docs in zip(gold['raw_docs'], pred['raw_docs']):
        gold_knowledge, pred_knowledge = {}, {}
        gold_knowledge['raw_docs'] = doc_concate(gold_docs)
        pred_knowledge['raw_docs'] = doc_concate(pred_docs)
        score += compare_two_answer(gold_knowledge, pred_knowledge)['raw_docs']
    return {'raw_docs': score / len(gold['raw_docs'])} 

def doc_filter_compare(gold, pred):
    score = 0
    for gold_doc_str, pred_doc_str in zip(gold['documents'], pred['documents']):
        score += compare_two_answer({'documents': gold_doc_str}, {'documents': pred_doc_str})['documents']
    return {'documents': score / len(gold['documents'])}

def sub_answer_generate(gold, pred):
    score = 0
    for gold_sub_answer, pred_sub_answer in zip(gold['sub_answers'], pred['sub_answers']):
        score += compare_two_answer({'sub_answers': gold_sub_answer}, {'sub_answers': pred_sub_answer})['sub_answers']
    return {'sub_answers': score / len(gold['sub_answers'])}

module_2_metric = {
    decompose_module.name: query_decompose_compare,
    hyde_module.name: hyde_retrieve_compare,
    doc_filter_module.name: doc_filter_compare,
    generator_module.name: sub_answer_generate,
    answer_compose_module.name: compare_two_answer,
}
from self_eval import evaluate_rag_answer_compatible

ms_boot = BootStrapLMSelection(
    workflow=rag_workflow,
    teachers='gpt-4o',
    module_2_options=lm_options,
    module_2_metric=module_2_metric,
    final_output_metric=evaluate_rag_answer_compatible,
    trainset_input=[state],
    trainset_label=None,
    max_sample_to_keep=4,
)


solutions = ms_boot.compile(
    log_dir='rag_compile_log_with_self_rag',
    gap=0.0,
)

rag_workflow.update_token_usage_summary()
pprint(rag_workflow.token_usage_buffer)