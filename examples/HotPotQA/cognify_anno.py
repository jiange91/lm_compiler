import dspy
from dsp.utils.utils import deduplicate
import dspy.evaluate
from compiler.utils import load_api_key
import string
import time

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

colbert = dspy.ColBERTv2(url='http://192.168.1.16:8893/api/search')
dspy.configure(rm=colbert)

import copy
from compiler.cog_hub.reasoning import ZeroShotCoT
from compiler.cog_hub.common import NoChange
from compiler.llm.model import LMConfig, CogLM
from compiler.llm import InputVar, OutputLabel
from compiler.frontends.dspy.connector import as_predict
from evaluator import answer_f1
from compiler.optimizer import register_opt_program_entry

lm_config = LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

initial_query_prompt = """
You are an expert at crafting precise search queries based on a provided question. Your sole task is to generate a detailed and well-structured search query that will help retrieve relevant external documents containing information needed to answer the question.

You should not answer the question directly, nor assume any prior knowledge. Instead, focus on constructing a search query that explicitly seeks external sources of information and considers the question's key elements, context, and possible nuances. Think carefully about the implications of your search and ensure that the search query encapsulates the key elements needed to retrieve the most pertinent information.
"""
first_query_agent = CogLM(agent_name="generate_query",
                          system_prompt=initial_query_prompt,
                          input_variables=[InputVar(name="question")],
                          output=OutputLabel(name="search_query", custom_output_format_instructions="Output only the search query, without any prefixes, or additional text."),
                          lm_config=lm_config)

following_query_prompt = """
You are good at extract relevant details from the provided context and question. Your task is to propose an effective search query that will help retrieve additional information to answer the question. Think carefully about the implications of your search. The search query should target the missing information while avoiding redundancy. 

You should not answer the question directly, nor assume any prior knowledge. You must generate an accurate search query that considers the context and question to retrieve the most relevant information.
"""

following_query_agent = CogLM(agent_name="refine_query",
                          system_prompt=following_query_prompt,
                          input_variables=[InputVar(name="context"), InputVar(name="question")],
                          output=OutputLabel(name="search_query", custom_output_format_instructions="Output only the search query, without any prefixes, or additional text."),
                          lm_config=lm_config)

answer_prompt = """
You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question.
"""
answer_agent = CogLM(agent_name="generate_answer",
                     system_prompt=answer_prompt,
                     input_variables=[InputVar(name="context"), InputVar(name="question")],
                     output=OutputLabel(name="answer", custom_output_format_instructions="Output the answer directly without unnecessary details, explanations, or repetition."),
                     lm_config=lm_config)

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.initial_generate_query = as_predict(first_query_agent)
        self.following_generate_query = as_predict(following_query_agent)
        self.generate_answer = as_predict(answer_agent)
    
    def doc_str(self, context):
        docs = []
        for i, c in enumerate(context):
            docs.append(f"[{i+1}]: {c}")
        return "\n".join(docs)

    def forward(self, question):
        context = []

        search_query = self.initial_generate_query(question=question).search_query
        # avoid only searching the first line
        search_query = search_query.replace("\n", ". ")
        print(search_query)
        passages = self.retrieve(search_query).passages
        context = deduplicate(context + passages)
        print(self.doc_str(context))
        
        search_query = self.following_generate_query(context=self.doc_str(context), question=question).search_query
        # avoid only searching the first line
        search_query = search_query.replace("\n", ". ")
        print(search_query)
        passages = self.retrieve(search_query).passages
        context = deduplicate(context + passages)
        print(self.doc_str(context))
        
        answer = self.generate_answer(context=self.doc_str(context), question=question).answer
        return answer

pipeline = BasicMH(passages_per_hop=2)

@register_opt_program_entry
def do_qa(question: str):
    answer = pipeline(question=question)
    return answer

if __name__ == "__main__":
    input = "What was the 2010 population of the birthplace of Gerard Piel?"
    answer = do_qa(input)
    label = '17,121'
    print(f'Answer: {answer}')
    print(f'Score: {answer_f1(label, answer)}')
    
    price = 0.0
    for m in [first_query_agent, following_query_agent, answer_agent]:
        price += m.get_total_cost()
    print(f'Token price: {price}')