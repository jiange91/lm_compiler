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

colbert = dspy.ColBERTv2(url='http://192.168.1.18:8893/api/search')

dspy.configure(lm=gpt4o_mini, rm=colbert)

# dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
# trainset = [x.with_inputs('question') for x in dataset.train[0:100]]
# valset = [x.with_inputs('question') for x in dataset.train[100:150]]
# devset = [x.with_inputs('question') for x in dataset.dev]

# print(devset[0])
# devset = devset[:50]

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
agent.load('/mnt/ssd4/lm_compiler/examples/HotPotQA/optimized_qa.dspy')

# print(agent.generate_query[0].extended_signature)

from cognify_anno import answer_f1

trainset = [
    ("""Are Walt Disney and Sacro GRA both documentry films?""", """yes"""),
    ("""What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?""", """design their own interdisciplinary program"""),
    ("""Which is published more frequently, The People's Friend or Bust?""", """The People's Friend"""),
    ("""How much is spent on the type of whiskey that 1792 Whiskey is in the United States?""", """about $2.7 billion"""),
    ("""The place where John Laub is an American criminologist and Distinguished University Professor in the Department of Criminology and Criminal Justice at was founded in what year?""", """1856"""),
    ("""What year did the mountain known in Italian as "Monte Vesuvio", erupt?""", """79 AD"""),
    ("""What was the full name of the author that memorialized Susan Bertie through her single volume of poems?""", """Emilia Lanier"""),
    ("""How many seasons did, the Guard with a FG%% around .420, play in the NBA ?""", """14 seasons"""),
    ("""Estonian Philharmonic Chamber Choir won the grammy Award for Best Choral Performance for two songs by a composer born in what year ?""", """1935"""),
    ("""Which of the sport analyst of The Experts Network is nicknamed  "The Iron Man"?""", """Calvin Edwin Ripken Jr."""),
    ("""What are both National Bird and America's Heart and Soul?""", """What are both National Bird and America's Heart and Soul?"""),
    ("""What was the 2010 population of the birthplace of Gerard Piel?""", """17,121"""),
    ("""On what streets is the hospital that cared for Molly Meldrum located?""", """the corner of Commercial and Punt Roads"""),
]

scores = []
for datapoint in trainset[-2:-1]:
    print("Question:", datapoint[0])
    question, answer = datapoint
    prediction = agent(question=question).answer
    gpt4o_mini.inspect_history(3)
    score = answer_f1(answer, prediction)
    scores.append(score)

# for datapoint in devset:
#     print("Question:", datapoint.question)
#     question, answer = datapoint.question, datapoint.answer
#     prediction = agent(question=question).answer
#     score = answer_f1(answer, prediction)
#     scores.append(score)
    
print("Avg score:", sum(scores) / len(scores))
