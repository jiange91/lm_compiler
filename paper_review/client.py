from compiler.utils import load_api_key, get_bill
from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import Retriever
from compiler.langchain_bridge.interface import LangChainLM
import copy
import json
import pprint

load_api_key('secrets.toml')

from grade_kernel import *

# ------------------------------------------------
# Original workflow
# ------------------------------------------------
# """
grade_module = ScientistLM('grade', grade_kernel)

review_workflow = Workflow()
review_workflow.add_module(grade_module)
review_workflow.set_exit_point(grade_module, 'review')

grade_module.lm_config = {'model': 'gpt-4o-mini'}

state = StatePool()
base_path = '/mnt/ssd4/AI-Scientist/review_iclr_bench/iclr_parsed/'
paper_txt = open(base_path + 'CC-BbehJKTe' + '.txt').read()
state.publish({'paper_text': paper_txt})

review_workflow.run(state)
print(state.news('review'))
# """
# ------------------------------------------------
# Modified workflow
# ------------------------------------------------
"""
summary_module = LangChainLM('summary', generate_summary)
strength_weakness_module = LangChainLM('strength_weakness', strength_weakness_kernel)
clarification_module = LangChainLM('clarification', clarification_kernel)
limitation_ethic_module = LangChainLM('limitation_ethic', limitation_n_ethical_kernel)
rating_decision_module = LangChainLM('rating_decision', rating_n_decision_kernel)

review_workflow = Workflow()
review_workflow.add_module(summary_module)
review_workflow.add_module(strength_weakness_module)
review_workflow.add_module(clarification_module)
review_workflow.add_module(limitation_ethic_module)
review_workflow.add_module(rating_decision_module)

review_workflow.add_edge(strength_weakness_module, limitation_ethic_module)
review_workflow.add_edge(summary_module, rating_decision_module)
review_workflow.add_edge(strength_weakness_module, rating_decision_module)
review_workflow.add_edge(clarification_module, rating_decision_module)
review_workflow.add_edge(limitation_ethic_module, rating_decision_module)
review_workflow.set_exit_point(rating_decision_module, 'rating_decision')

openai_kwargs = {
    'temperature': 0,
}

sample_lm = 'gpt-4o-mini'
summary_module.lm_config = {'model': sample_lm, **openai_kwargs}
strength_weakness_module.lm_config = {'model': sample_lm, **openai_kwargs}
clarification_module.lm_config = {'model': sample_lm, **openai_kwargs}
limitation_ethic_module.lm_config = {'model': sample_lm, **openai_kwargs}
rating_decision_module.lm_config = {'model': sample_lm, **openai_kwargs}
"""

"""
import pandas as pd
wrong_pred_df = pd.read_csv('/mnt/ssd4/AI-Scientist/review_iclr_bench/llm_reviews/4o-fewshot-1-reflect-5-ensemble-5-wrong_pred.csv', index_col="paper_id")
# num_llm_reviews = wrong_pred_df.shape[0]
num_llm_reviews = 30

new_llm_reviews = {}
correct = 0
still_wrong = []
for i in range(num_llm_reviews):
    paper_id = wrong_pred_df.iloc[i].name

    state = StatePool()
    base_path = '/mnt/ssd4/AI-Scientist/review_iclr_bench/iclr_parsed/'
    paper_txt = open(base_path + paper_id + '.txt').read()
    state.publish({'paper_text': paper_txt})

    review_workflow.reset_modules()
    review_workflow.run(state)
    new_llm_reviews[paper_id] = state.news('rating_decision')
    # append to log file
    with open('llm_review_log.json', 'a') as f:
        json.dump({paper_id: state.news('rating_decision')}, f)
        f.write('\n')
    decision = 'Accept' if 'Congratulations' in state.news('rating_decision') else 'Reject'
    
    if wrong_pred_df["Decision"].loc[paper_id] != decision:
        correct += 1
    else:
        still_wrong.append(paper_id) 

print(f"acc: {correct}/{num_llm_reviews}")
json.dump(new_llm_reviews, open('new_llm_reviews.json', 'w'))
json.dump(still_wrong, open('still_wrong.json', 'w'))

state = StatePool()
base_path = '/mnt/ssd4/AI-Scientist/review_iclr_bench/iclr_parsed/'
paper_txt = open(base_path + 'CC-BbehJKTe' + '.txt').read()
state.publish({'paper_text': paper_txt})
review_workflow.run(state)
    
pprint.pprint(state.news('rating_decision'))
# review_workflow.log_token_usage('token_usage.json')
# print(get_bill(review_workflow.token_usage_buffer))
"""