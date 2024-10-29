import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

gpt4o_mini = dspy.LM('gpt-4o-mini', max_tokens=1000)

colbert = dspy.ColBERTv2(url='http://192.168.1.16:8893/api/search')

dspy.configure(lm=gpt4o_mini, rm=colbert)

from dsp.utils.utils import deduplicate

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []

        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            print("Search query:", search_query)
            passages = self.retrieve(search_query).passages
            print("Passages:", passages)
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question).copy(context=context)
        return answer
    
agent = BasicMH(passages_per_hop=2)

