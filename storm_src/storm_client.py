import dspy
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import os
from pathlib import Path
import pandas as pd

from persona_generator import CreateWriterWithPersona
from knowledge_curation import StormKnowledgeCurationModule
from outline_generation import WriteOutline
from article_generation import StormArticleGenerationModule
from article_polish import StormArticlePolishingModule
from retriever import StormRetriever
from rm import YouRM, BingSearch
from common import compare_two_answer, StormState
from utils import topic_dir, data_loader_article, preprocess_text

from compiler.IR.program import Workflow, Module, StatePool
from compiler.dspy_bridge.interface import DSPyLM
from compiler.optimizer.bootstrap import BootStrapLMSelection
from compiler.IR.modules import Input
from compiler.utils import load_api_key
import logging

# logging.basicConfig(level=logging.INFO)


# --------------------------------------------
# Compiler Client
# --------------------------------------------
load_api_key('secrets.toml')

# Get persona writer module
get_personas = CreateWriterWithPersona()

# Reseach Knowledge Conversation Module
# rm = YouRM(ydc_api_key=os.getenv('YDC_API_KEY'), k=3)
rm = BingSearch(k=3)
knowledge_curation = StormKnowledgeCurationModule(
    retriever=StormRetriever(rm=rm, k=3),
    max_search_queries_per_turn=3,
    search_top_k=3,
    max_conv_turn=1,
    max_thread_num=5,
)

# Write outline
write_ouline = WriteOutline()
# Write article draft
article_generation = StormArticleGenerationModule(retrieve_top_k=3, max_thread_num=5)
# Polish article
article_polish = StormArticlePolishingModule()

# user_input_module = Input({
#     'topic': 'Taylor Hawkins',
#     'max_perspective': 5,
# })
find_related_topic_module = DSPyLM('find_related_topics', get_personas.agent_get_topics)
get_personas_module = DSPyLM('get_personas', get_personas.agent_get_personas)
knowledge_curation_module = DSPyLM('knowledge_curation', knowledge_curation.research_kernel)
outline_draft_module = DSPyLM('outline_draft', write_ouline.generate_draft_outline)
outline_refine_module = DSPyLM('outline_refine', write_ouline.refine_outline)
draft_article_module = DSPyLM('draft_article', article_generation.generate_article_kernel)
polish_article_module = DSPyLM('polish_article', article_polish.polish_article_kernel)

storm_workflow = Workflow()
# storm_workflow.add_module(user_input_module)
storm_workflow.add_module(find_related_topic_module)
storm_workflow.add_module(get_personas_module)
storm_workflow.add_module(knowledge_curation_module)
storm_workflow.add_module(outline_draft_module)
storm_workflow.add_module(outline_refine_module)
storm_workflow.add_module(draft_article_module)
storm_workflow.add_module(polish_article_module)

# storm_workflow.add_edge(user_input_module, find_related_topic_module)
# storm_workflow.add_edge(user_input_module, outline_draft_module)
storm_workflow.add_edge(find_related_topic_module, get_personas_module)
storm_workflow.add_edge(get_personas_module, knowledge_curation_module)
storm_workflow.add_edge(knowledge_curation_module, outline_refine_module)
storm_workflow.add_edge(outline_draft_module, outline_refine_module)
storm_workflow.add_edge(outline_refine_module, draft_article_module)
storm_workflow.add_edge(draft_article_module, polish_article_module)
storm_workflow.set_exit_point(polish_article_module, 'polished_article')

openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'api_provider': os.getenv('OPENAI_API_TYPE'),
    'temperature': 1.0,
    'top_p': 0.9,
}

sample_lm = 'gpt-4o-mini'
find_related_topic_module.lm_config = {'model': sample_lm, 'max_tokens': 500, **openai_kwargs}
get_personas_module.lm_config = {'model': sample_lm, 'max_tokens': 500, **openai_kwargs}
knowledge_curation_module.lm_config = {'model': sample_lm, 'max_tokens': 500, **openai_kwargs}
outline_draft_module.lm_config = {'model': sample_lm, 'max_tokens': 400, **openai_kwargs}
outline_refine_module.lm_config = {'model': sample_lm, 'max_tokens': 400, **openai_kwargs}
draft_article_module.lm_config = {'model': sample_lm, 'max_tokens': 700, **openai_kwargs}
polish_article_module.lm_config = {'model': sample_lm, 'max_tokens': 4000, **openai_kwargs}


# Initialize State
trainset_input: list[StatePool] = []
trainset_label: list[str] = []
for topic, url, golden_answer in data_loader_article('storm_src/trainset.csv'):
    state = StormState()
    state.publish({
        'topic': topic,
        'max_perspective': 5,
        'ground_truth_url': url,
    })
    trainset_input.append(state)
    trainset_label.append(golden_answer)
    Path(topic_dir(state.news('topic'))).mkdir(parents=True, exist_ok=True)


# --------------------------------------------
# Sample run
# --------------------------------------------
storm_workflow.run(state=state)
print(state.state)
exit()

# --------------------------------------------
# Bootstrap
# --------------------------------------------

lm_options = [
    'gpt-4o-mini', 
    'gpt-4o',
]

def final_output_metric(gold: dict, pred: dict):
    gold['final_output'] = preprocess_text(gold['final_output'])
    pred['final_output'] = preprocess_text(pred['final_output'])
    return compare_two_answer(gold, pred)
    

ms_boot = BootStrapLMSelection(
    workflow=storm_workflow,
    teachers='gpt-4o',
    module_2_options=lm_options,
    module_2_metric=compare_two_answer,
    final_output_metric=final_output_metric,
    trainset_input=trainset_input,
    trainset_label=trainset_label,
    max_sample_to_keep=4,
)

# ms_boot.compile(
#     trainset=[state], 
#     label_path='labels-4o.json',
#     profile_path='module_option_profile.json',
#     curve_dir='storm_curve',
# )

solutions = ms_boot.compile(
    label_path='compile_log/labels-4o.json',
    profile_path='compile_log/module_option_profile.json',
    curve_path='compile_log/storm_curve.joblib',
    solution_path='compile_log/solutions_95.json',
)
