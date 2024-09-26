import dspy
from dsp.utils.utils import deduplicate
import dspy.evaluate
from compiler.utils import load_api_key
import string

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

colbert = dspy.ColBERTv2(url='http://192.168.1.18:8893/api/search')
dspy.configure(rm=colbert)

from compiler.optimizer.params.reasoning import ZeroShotCoT
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn

lm_config = {
    'model': 'gpt-4o-mini',
    'temperature': 0.0,
}

initial_query_prompt = """
Generate a search query based on the provided question. The query will be used for retrieving documents so it should identify key entities relevant to the inquiry.
"""
first_query_semantic = LangChainSemantic(
    system_prompt=initial_query_prompt,
    inputs=['question'],
    output_format='search_query',
)
first_query_agent = LangChainLM('generate_query', first_query_semantic, opt_register=True)
first_query_agent.lm_config = lm_config

following_query_prompt = """
Given the context and question, create a search query that accurately reflects the main focus of the question while utilizing the relevant information from the context.
"""
following_query_semantic = LangChainSemantic(
    system_prompt=following_query_prompt,
    inputs=['context', 'question'],
    output_format='search_query',
)
following_query_agent = LangChainLM('refine_query', following_query_semantic, opt_register=True)
following_query_agent.lm_config = lm_config

answer_prompt = """
Given the context and question, analyze the information and produce a concise and clear `answer`. Ensure that the answer is grounded by the context and directly addresses the question asked. 
"""
answer_semantic = LangChainSemantic(
    system_prompt=answer_prompt,
    inputs=['context', 'question'],
    output_format='answer',
)
answer_agent = LangChainLM('generate_answer', answer_semantic, opt_register=True)
answer_agent.lm_config = lm_config

cot_fixed = False
if cot_fixed:
    ZeroShotCoT.direct_apply(first_query_agent)
    ZeroShotCoT.direct_apply(following_query_agent)
    ZeroShotCoT.direct_apply(answer_agent)


class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.initial_generate_query = first_query_agent.as_runnable()
        self.follwing_generate_query = following_query_agent.as_runnable()
        self.generate_answer = answer_agent.as_runnable()

    def forward(self, question):
        context = []

        search_query = self.initial_generate_query.invoke({'question': question}).content
        passages = self.retrieve(search_query).passages
        context = deduplicate(context + passages)
        
        for _ in range(2-1):
            search_query = self.follwing_generate_query.invoke({'context': context, 'question': question}).content

        answer = self.generate_answer.invoke({'context': context, 'question': question}).content
        return answer

qa_agent = BasicMH()

@register_opt_program_entry
def trial(question: str):
    return qa_agent(question=question)

from dsp.utils.metrics import HotPotF1, F1
dspy.evaluate.answer_exact_match
@register_opt_score_fn
def answer_f1(label: str, pred: str):
    print(f'Label: {label}')
    print(f'Pred: {pred}')
    if isinstance(label, str):
        label = [label]
    return F1(pred, label)

if __name__ == "__main__":
    input = {'question': "Are both Cangzhou and Qionghai in the Hebei province of China?"}
    answer = trial(input)
    label = 'no'
    print(f'Answer: {answer}')
    print(f'Score: {answer_f1(label, answer)}')