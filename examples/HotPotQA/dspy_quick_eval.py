import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

gpt4o_mini = dspy.OpenAI('gpt-4o-mini', max_tokens=1000)
colbert = dspy.ColBERTv2(url='http://192.168.1.18:8893/api/search')
dspy.configure(lm=gpt4o_mini, rm=colbert)

dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
trainset = [x.with_inputs('question') for x in dataset.train[0:100]]
valset = [x.with_inputs('question') for x in dataset.train[100:150]]
devset = [x.with_inputs('question') for x in dataset.dev]

# show an example datapoint; it's just a question-answer pair
print(devset[0])

devset = devset[:50]

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

        return self.generate_answer(context=context, question=question).copy(context=context)
    
agent = BasicMH(passages_per_hop=2)
agent.load('optimized_qa.dspy')

# print(agent.generate_query[0].extended_signature)

from cognify_anno import answer_f1

scores = []
for datapoint in devset:
    print("Question:", datapoint.question)
    question, answer = datapoint.question, datapoint.answer
    prediction = agent(question=question).answer
    score = answer_f1(answer, prediction)
    scores.append(score)
    
print("Avg score:", sum(scores) / len(scores))