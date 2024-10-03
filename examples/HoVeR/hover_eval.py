import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

gpt4o_mini = dspy.OpenAI('gpt-4o-mini', max_tokens=1000)
colbert = dspy.ColBERTv2(url='http://192.168.1.18:8893/api/search')
dspy.configure(lm=gpt4o_mini, rm=colbert)

import json
import pandas as pd

class CustomDataset:
    def __init__(self, train_seed=None, train_size=150, eval_seed=None, dev_size=200, test_size=0):
        self.data_dir = 'examples/HoVeR/hover_data/hover'
        self.train_data = self.load_data('train', train_size)
        self.dev_data = self.load_data('dev', dev_size)
        self.test_data = self.load_data('test', test_size)

    def load_data(self, split, size):
        # Load JSON data
        json_path = f'{self.data_dir}/{split}/qas.json'
        data = []
        with open(json_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))

        # Load TSV data
        tsv_path = f'{self.data_dir}/{split}/questions.tsv'
        questions = pd.read_csv(tsv_path, sep='\t', header=None)
        questions.columns = ['id','question']

        # Combine JSON and questions based on your JSON structure
        for i, item in enumerate(data):
            item['question'] = questions.loc[i, 'question']
            if i >= size - 1:
                break

        return data[:size]


dataset = CustomDataset(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
trainset = dataset.train_data[0:100]
valset = dataset.train_data[100:150]
devset = dataset.dev_data[:50]

class RetrieveMultiHop( dspy.Module ) :
    def __init__( self ) :
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim ,summary_1 -> query ")
        self.create_query_hop3 = dspy.ChainOfThought("claim ,summary_1 , summary_2 -> query ")
        self.retrieve_k = dspy.Retrieve( k = self.k )
        self.summarize1 = dspy.ChainOfThought("claim , passages ->summary ")
        self.summarize2 = dspy.ChainOfThought("claim , context ,passages -> summary ")

    def forward( self , claim ) :
        # HOP 1
        hop1_docs = self.retrieve_k( claim ).passages
        summary_1 = self.summarize1( claim = claim , passages =hop1_docs ).summary # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2( claim = claim ,summary_1 = summary_1 ).query
        hop2_docs = self.retrieve_k( hop2_query ).passages
        summary_2 = self.summarize2(claim = claim , context =summary_1 , passages = hop2_docs ).summary

        # HOP 3
        hop3_query = self.create_query_hop3( claim = claim ,summary_1 = summary_1 , summary_2 = summary_2 ).query
        hop3_docs = self.retrieve_k( hop3_query ).passages

        return dspy.Prediction( retrieved_docs = hop1_docs + hop2_docs + hop3_docs )

agent = RetrieveMultiHop()
# agent.load('examples/HotPotQA/optimized_qa.dspy')

# print(agent.generate_query[0].extended_signature)

print(devset[0])

from collections import Counter
from dsp.utils.metrics import normalize_text

def f1_score(prediction, ground_truth):

    
    ground_truth_tokens = ground_truth
    prediction_tokens = [str(pred.split(" | ")[0]).strip().strip('\'"')  for pred in prediction]
        
    print(ground_truth_tokens,prediction_tokens)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    
    # gt_in_pred = 0
    # for pred in prediction_tokens:
    #     if pred in ground_truth_tokens:
    #         gt_in_pred += 1
    # print(gt_in_pred)

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    print(num_same)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

scores = []
for datapoint in devset[:1]:
    print("Question:", datapoint['question'])
    question= datapoint['question']
    prediction = agent(claim=question).retrieved_docs
    # print(prediction)
    score = f1_score(prediction, datapoint['support_titles'])
    scores.append(score)
    
print("Avg score:", sum(scores) / len(scores))