import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dsp.utils.utils import deduplicate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

colbert = dspy.ColBERTv2(url='http://192.168.1.16:8893/api/search') # Change to your ColBERT endpoint
dspy.configure(rm=colbert)

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import LMConfig, LLMPredictor, Demonstration, TokenUsage

dummy_lm_config = LMConfig(
    provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
        'max_tokens': 1000,
    }
)

#===============================================================================
# Cognify Annotations
# Prompts are generated by asking GPT-4o only once
#===============================================================================
summarize1_system_prompt = """
You are a research analyst specializing in fact-checking complex claims through evidence gathering.

You will be given a `claim` requiring verification and a set of `passages` retrieved from the document search. As a research analyst, your goal is to examine these passages carefully and summarize key points that relate directly to the claim. Highlight essential details, contextual insights, and any information that may either support or provide background on the claim. Your summary should focus on clarity and relevance, setting the stage for deeper investigation.

Please form your summary carefully. It will act as the foundational context for the next document search query.
"""
summarize1_semantic = LangChainSemantic(
    system_prompt=summarize1_system_prompt,
    inputs=['claim', 'passages'],
    output_format='summary',
)
summarize1__agent = LangChainLM('summarize1', summarize1_semantic, opt_register=True)
summarize1__agent.lm_config = dummy_lm_config
#===============================================================================
create_query_hop2_system_prompt = """
You are a strategic search specialist skilled in crafting precise queries to uncover additional evidence.

You will be given a `claim` that needs to be explored, a `summary` of the relevant information retrieved related to the claim. Your task is to generate a focused and clear query that will help retrieve more relevant external documents. This query should aim to address gaps, ambiguities, or details missing in the existing information. Target specific information or clarifications that could strengthen the evidence for or against the claim.
"""
create_query_hop2_semantic = LangChainSemantic(
    system_prompt=create_query_hop2_system_prompt,
    inputs=['claim', 'summary'],
    output_format='query',
    output_format_instructions="Output only the search query, without any prefixes, or additional text."
)
create_query_hop2_agent = LangChainLM('create_query_hop2', create_query_hop2_semantic, opt_register=True)
create_query_hop2_agent.lm_config = dummy_lm_config
#===============================================================================
summarize_2_system_prompt = """
You are an evidence synthesis expert specializing in extracting distinct, complementary insights to deepen understanding of claims.

You will be given a `claim` requiring verification, a summarized `context` of the existing information, and a set of `passages` newly retrieved from the document search.

Your role is to summarize the new passages. Summarize the information clearly, emphasizing any new details that support, refute, or add depth to the claim. Your summary should provide unique and complementary knowledge base that is not covered in the provided context to advance the understanding of the claim.
"""
summarize_2_semantic = LangChainSemantic(
    system_prompt=summarize_2_system_prompt,
    inputs=['claim', 'context', 'passages'],
    output_format='summary',
)
summarize2__agent = LangChainLM('summarize2', summarize_2_semantic, opt_register=True)
summarize2__agent.lm_config = dummy_lm_config
#===============================================================================
create_query_hop3_system_prompt = """
You are an investigative researcher focused on constructing comprehensive queries to gather conclusive evidence.

You will be given a `claim` requiring verification, two `summary`s of the relevant information retrieved related to the claim. Your task is to formulate a final, detailed query that will help retrieve more relevant external documents. This query should target any remaining evidence gaps needed to fully verify or refute the claim. Focus on details and unresolved elements that will bring the verification process to a comprehensive conclusion.
"""
create_query_hop3_semantic = LangChainSemantic(
    system_prompt=create_query_hop3_system_prompt,
    inputs=['claim', 'summary_1', 'summary_2'],
    output_format='query',
    output_format_instructions="Output only the search query, without any prefixes, or additional text."
)
create_query_hop3_agent = LangChainLM('create_query_hop3', create_query_hop3_semantic, opt_register=True)
create_query_hop3_agent.lm_config = dummy_lm_config
#===============================================================================

class RetrieveMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = create_query_hop2_agent.as_runnable()
        self.create_query_hop3 = create_query_hop3_agent.as_runnable()
        self.retrieve_k = dspy.Retrieve(k=self.k)
        self.summarize1 = summarize1__agent.as_runnable()
        self.summarize2 = summarize2__agent.as_runnable()

    def forward(self, claim):
        # HOP 1
        hop1_docs = self.retrieve_k(claim, with_metadata=True)
        summary_1 = self.summarize1.invoke({'claim': claim, 'passages': hop1_docs.passages}).content # Summarize top k docs

        # HOP 2
        hop2_query = self.create_query_hop2.invoke({'claim': claim, 'summary': summary_1}).content
        hop2_docs = self.retrieve_k(hop2_query, with_metadata=True)
        summary_2 = self.summarize2.invoke({'claim': claim, 'context': summary_1, 'passages': hop2_docs.passages}).content

        # HOP 3
        hop3_query = self.create_query_hop3.invoke({'claim': claim, 'summary_1': summary_1, 'summary_2': summary_2}).content
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