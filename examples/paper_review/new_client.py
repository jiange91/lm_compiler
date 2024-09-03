from compiler.utils import load_api_key, get_bill
from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import Retriever, Input, Output
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import os
import copy
import json
import pprint

load_api_key('secrets.toml')

from perform_review import reviewer_system_prompt_neg, neurips_form, get_review_fewshot_examples

# ------------------------------------------------
# Original workflow
# ------------------------------------------------
# """
class ReviewSchema(BaseModel):
    "Paper review schema"
    summary: str = Field(description='A summary of the paper content and its contributions.')
    strengths: List[str] = Field(description='A list of strengths of the paper.')
    weaknesses: List[str] = Field(description='A list of weaknesses of the paper.')
    originality: int = Field(description='A rating from 1 to 4 (low, medium, high, very high)')
    quality: int = Field(description='A rating from 1 to 4 (low, medium, high, very high)') 
    clarity: int = Field(description='A rating from 1 to 4 (low, medium, high, very high).') 
    significance: int = Field(description='A rating from 1 to 4 (low, medium, high, very high).') 
    questions: List[str] = Field(description='A list of clarifying questions to be answered by the paper authors.') 
    limitations: List[str] = Field(description='A list of limitations and potential negative societal impacts of the work.') 
    ethical_concerns: bool = Field(description='A boolean value (true or false) indicating whether there are ethical concerns.') 
    soundness: int = Field(description='A rating from 1 to 4 (poor, fair, good, excellent).') 
    presentation: int = Field(description='A rating from 1 to 4 (poor, fair, good, excellent).') 
    contribution: int = Field(description='A rating from 1 to 4 (poor, fair, good, excellent).') 
    overall: int = Field(description='A rating from 1 to 10 (very strong reject to award quality).') 
    confidence: int = Field(description='A rating from 1 to 5 (low, medium, high, very high, absolute).') 
    decision: str = Field(description='A decision that has to be one of the following: Accept, Reject.') 

system_prompt = reviewer_system_prompt_neg + \
    "\nBelow is Neurips conference review form, please follow the instructions when giving your review\n\n" + \
    neurips_form


review_semantic = LangChainSemantic(
    system_prompt=system_prompt,
    inputs=['paper_text'],
    output_format=ReviewSchema,
)

reviewer_module = LangChainLM('reviewer', review_semantic)

review_workflow = Workflow('review paper flow')
review_workflow.add_module(reviewer_module)
review_workflow.add_module(Input('paper_input'))
review_workflow.add_module(Output('review_output'))
review_workflow.add_edge('paper_input', 'reviewer')
review_workflow.add_edge('reviewer', 'review_output')
review_workflow.compile()

openai_kwargs = {
    'temperature': 0,
}

reviewer_module.lm_config = {'model': 'gpt-4o-2024-05-13', **openai_kwargs}
# reviewer_module.lm_config = {'model': 'gpt-4o', **openai_kwargs}

base_path = '/mnt/ssd4/AI-Scientist/review_iclr_bench/iclr_parsed/'

def sample_run(paper_id: str):
    paper_txt = open(base_path + paper_id + '.txt').read()
    state = StatePool()
    state.init({'paper_text': paper_txt})

    review_workflow.reset()
    review_workflow.pregel_run(state)
    print(json.dumps(state.all_news(excludes=['paper_text']), indent=4))
    return state.news('decision')

log_dir = 'examples/paper_review/compile_logs'
from compiler.optimizer.decompose import LMTaskDecompose

def task_disambiguous():
    decomposer = LMTaskDecompose(
        workflow=review_workflow,
    )
    decomposer.decompose(
        log_dir=log_dir,
        threshold=3,
    )

# task_disambiguous()
# exit()

# rej: XJFGyJEBLuz
# acc_poster: CVfLvQq9gLo
# acc_oral: EhYjZy6e1gJ
# acc_splotlight: 7gWSJrP3opB
sample_run('7gWSJrP3opB')
exit()

from iclr_analysis import prep_open_review_data
import pandas as pd
ore_ratings: pd.DataFrame = prep_open_review_data("/mnt/ssd4/AI-Scientist/review_iclr_bench/ratings_subset.tsv")

def bench(n):
    results = {}
    for paper_id, row in ore_ratings.head(n).iterrows():
        print(f"Reviewing Paper: {paper_id}")
        decision = sample_run(paper_id)
        print(f"Decision: {decision}")
        print(f"True decision: {row['simplified_decision']}")
        results[paper_id] = {'decision': decision, 'true_decision': row['simplified_decision']}
    acc = sum([1 for k, v in results.items() if v['decision'] == v['true_decision']]) / len(results)
    print(f"Accuracy: {acc}")
    return results

results = bench(30)
json.dump(results, open(os.path.join(log_dir, 'new_results.json'), 'w+'), indent=4)

def bench_wrong_preds():
    wrong_pred_df = pd.read_csv('/mnt/ssd4/AI-Scientist/review_iclr_bench/llm_reviews/4o-fewshot-1-reflect-5-ensemble-5-wrong_pred.csv', index_col="paper_id")
    # num_llm_reviews = wrong_pred_df.shape[0]
    num_llm_reviews = 30

    new_llm_reviews = {}
    correct = 0
    still_wrong = []
    for i in range(num_llm_reviews):
        paper_id = wrong_pred_df.iloc[i].name

        decision = sample_run(paper_id)
        new_llm_reviews[paper_id] = decision
        
        if wrong_pred_df["Decision"].loc[paper_id] != decision:
            correct += 1
        else:
            still_wrong.append(paper_id) 

    print(f"acc: {correct}/{num_llm_reviews}")
    json.dump(new_llm_reviews, open('new_llm_reviews.json', 'w'), indent=4)
    json.dump(still_wrong, open('still_wrong.json', 'w'))