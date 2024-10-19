import dspy
from dsp.utils.utils import deduplicate
import dspy.evaluate
from compiler.utils import load_api_key
import string
import time

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

colbert = dspy.ColBERTv2(url='http://192.168.1.18:8893/api/search')
dspy.configure(rm=colbert)

import copy
from compiler.optimizer.params.reasoning import ZeroShotCoT
from compiler.optimizer.params.common import IdentityOption
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration, TokenUsage
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn

qgen_lm_config = LMConfig(
    # provider='fireworks',
    # kwargs= {
    #     'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
    #     'temperature': 0.0,
    # }
    
    provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
    
    # provider="local",
    # model='llama-3.1-8b',
    # kwargs={
    #     'temperature': 0.0,
    #     'openai_api_base': 'http://192.168.1.16:30000/v1',
    # }
)

initial_query_prompt = """
You are an expert at crafting precise search queries based on a provided question. Your sole task is to generate a detailed and well-structured search query that will help retrieve relevant external documents containing information needed to answer the question.

You should not answer the question directly, nor assume any prior knowledge. Instead, focus on constructing a search query that explicitly seeks external sources of information and considers the question's key elements, context, and possible nuances. Think carefully about the implications of your search and ensure that the search query encapsulates the key elements needed to retrieve the most pertinent information.
"""
first_query_semantic = LangChainSemantic(
    system_prompt=initial_query_prompt,
    inputs=['question'],
    output_format='search_query',
    output_format_instructions='Output only the search query, without any prefixes, or additional text.'
)
first_query_agent = LangChainLM('generate_query', first_query_semantic, opt_register=True)
first_query_agent.lm_config = qgen_lm_config

following_query_prompt = """
You are in a critical situation where accurate information is essential for making informed decisions.

You are good at extract relevant details from the provided context and question. Your task is to propose an effective search query that will help retrieve additional information to answer the question. Think carefully about the implications of your search. The search query should target the missing information while avoiding redundancy. 

You should not answer the question directly, nor assume any prior knowledge. You must generate an accurate search query that considers the context and question to retrieve the most relevant information.
"""
following_query_semantic = LangChainSemantic(
    system_prompt=following_query_prompt,
    inputs=['context', 'question'],
    output_format='search_query',
    output_format_instructions='Output only the search query, without any prefixes, or additional text.'
)
following_query_agent = LangChainLM('refine_query', following_query_semantic, opt_register=True)
refine_lm_config = LMConfig(
    provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
    
    # provider="local",
    # model='llama-3.1-8b',
    # kwargs={
    #     'temperature': 0.0,
    #     'openai_api_base': 'http://192.168.1.16:30000/v1',
    # }
)
following_query_agent.lm_config = refine_lm_config

answer_prompt = """
You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question.
"""
answer_semantic = LangChainSemantic(
    system_prompt=answer_prompt,
    inputs=['context', 'question'],
    output_format='answer',
    output_format_instructions="Output the answer directly without unnecessary details, explanations, or repetition. Focus on delivering the key information in the most concise way possible (don't have to be a complete sentence)."
)
answer_agent = LangChainLM('generate_answer', answer_semantic, opt_register=True)
answer_lm_config = LMConfig(
    provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
    
    # provider="local",
    # model='llama-3.1-8b',
    # kwargs={
    #     'temperature': 0.0,
    #     'openai_api_base': 'http://192.168.1.16:30000/v1',
    # }
)
answer_agent.lm_config = answer_lm_config

cot_fixed = False
if cot_fixed:
    ZeroShotCoT.direct_apply(first_query_agent)
    ZeroShotCoT.direct_apply(following_query_agent)
    ZeroShotCoT.direct_apply(answer_agent)

    
detail_log_level = 1
if detail_log_level == 0:
    _print_internal = lambda *args, **kwargs: None
    _print_expo = lambda *args, **kwargs: None
elif detail_log_level == 1:
    _print_internal = lambda *args, **kwargs: None
    _print_expo = print
else:
    _print_internal = print
    _print_expo = print

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.initial_generate_query = first_query_agent.as_runnable()
        self.follwing_generate_query = following_query_agent.as_runnable()
        self.generate_answer = answer_agent.as_runnable()
    
    def doc_str(self, context):
        docs = []
        for i, c in enumerate(context):
            docs.append(f"[{i+1}]: {c}")
        return "\n".join(docs)

    def forward(self, question):
        context = []

        search_query = self.initial_generate_query.invoke({'question': question}).content
        # avoid only searching the first line
        search_query = search_query.replace("\n", ". ")
        _print_internal("Search query 1:", search_query)
        passages = self.retrieve(search_query).passages
        _print_internal("Passages 1:", passages)
        context = deduplicate(context + passages)
        
        for _ in range(2-1):
            search_query = self.follwing_generate_query.invoke({'context': self.doc_str(context), 'question': question}).content
            # avoid only searching the first line
            search_query = search_query.replace("\n", ". ")
            _print_internal("Search query 2:", search_query)
            passages = self.retrieve(search_query).passages
            _print_internal("Passages 2:", passages)
            context = deduplicate(context + passages)
        # _print_internal("Context:", context)
        answer = self.generate_answer.invoke({'context': self.doc_str(context), 'question': question}).content
        return answer

qa_agent = BasicMH(passages_per_hop=2)

@register_opt_program_entry
def trial(question: str):
    answer = qa_agent(question=question)
    _print_expo(f'Question: {question}')
    return answer

from dsp.utils.metrics import HotPotF1, F1
dspy.evaluate.answer_exact_match
@register_opt_score_fn
def answer_f1(label: str, pred: str):
    if isinstance(label, str):
        label = [label]
    score = F1(pred, label)
    _print_expo(f'Label: {label}')
    _print_expo(f'Pred: {pred}')
    _print_expo(f'Score: {score}\n')
    return score

from compiler.optimizer.params.reasoning import ZeroShotCoT
if __name__ == "__main__":
    input = "What was the 2010 population of the birthplace of Gerard Piel?"
    answer = trial(input)
    label = '17,121'
    print(f'Answer: {answer}')
    print(f'Score: {answer_f1(label, answer)}')
    
    price = 0.0
    for m in [first_query_agent, following_query_agent, answer_agent]:
        price += m.get_total_cost()
    print(f'Token price: {price}')