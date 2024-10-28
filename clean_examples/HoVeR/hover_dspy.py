import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dsp.utils.utils import deduplicate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

gpt4o_mini = dspy.LM('gpt-4o-mini', max_tokens=1000)
colbert = dspy.ColBERTv2(url='http://192.168.1.16:8893/api/search') # Change to your ColBERT endpoint
dspy.configure(lm=gpt4o_mini, rm=colbert)


class RetrieveMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("claim, summary_1 -> query")
        self.create_query_hop3 = dspy.ChainOfThought("claim, summary_1, summary_2 -> query")
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = dspy.ChainOfThought("claim, passages -> summary")
        self.summarize2 = dspy.ChainOfThought("claim, context, passages -> summary")

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim, with_metadata=True)
        summary_1 = self.summarize1(claim=claim, passages=hop1_docs.passages).summary  # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2(claim=claim, summary_1=summary_1).query
        hop2_docs = self.retrieve_k(hop2_query, with_metadata=True)
        summary_2 = self.summarize2(claim=claim, context=summary_1, passages=hop2_docs.passages).summary

        # HOP 3
        hop3_query = self.create_query_hop3(claim=claim, summary_1=summary_1, summary_2=summary_2).query
        hop3_docs = self.retrieve_k(hop3_query, with_metadata=True)

        scores, pids, passages = [], [], []
        
        for retrieval in [hop1_docs, hop2_docs, hop3_docs]:
            for score, pid, passage in zip(retrieval.score, retrieval.pid, retrieval.passages):
                scores.append(score)
                passages.append(passage)
                pids.append(pid)
        # use docs from top-10 passages
        sorted_passages = sorted(zip(scores, pids, passages), key=lambda x: x[0], reverse=True)[:10]
        scores, pids, passages = zip(*sorted_passages)
        return dspy.Prediction(scores=scores, pids=pids, passages=passages)


def doc_f1(gold, pred, trace=None):
    pred_pids = pred.pids
    # get f1 score for the retrieved docs
    gold_pids = set(gold.docs)
    pred_pids = set(pred_pids)
    common = gold_pids.intersection(pred_pids)
    precision = len(common) / len(pred_pids)
    recall = len(common) / len(gold_pids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return f1

if __name__ == "__main__":
    from data_loader import load_data
    train_set, val_set, dev_set = load_data()
    train_set = [dspy.Example(claim=claim, docs=docs).with_inputs("claim") for claim, docs in train_set]
    val_set = [dspy.Example(claim=claim, docs=docs).with_inputs("claim") for claim, docs in val_set]
    dev_set = [dspy.Example(claim=claim, docs=docs).with_inputs("claim") for claim, docs in dev_set]
    
    agent = RetrieveMultiHop()
    print(dev_set[1])
    pred = agent(claim = dev_set[1].claim)
    print(pred)
    pids = pred.pids
    print(pids)
    print(doc_f1(dev_set[1], pred))

