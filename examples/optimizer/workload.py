from compiler.utils import load_api_key, get_bill
from pydantic import BaseModel, Field

import copy
import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)


class ChessMove(BaseModel):
    """Response from the chess move task"""
    move: str = Field(
        description="The next move that will result in a checkmate"
    )
    
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic

semantic = LangChainSemantic(
    system_prompt="",
    inputs=["input"],
    output_format="move",
    following_messages=[
        HumanMessage("Given a series of chess moves written in Standard Algebraic Notation (SAN), give the next move that will result in a checkmate."), 
        HumanMessagePromptTemplate.from_template("moves:\n {input}\n")
    ]
)

qa_agent = LangChainLM('qa_agent', semantic)
qa_agent.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}

from compiler.IR.program import Workflow, Input, Output, StatePool

qa_flow = Workflow('qa_flow')

qa_flow.add_module(Input('start'))
qa_flow.add_module(Output('end'))
qa_flow.add_module(qa_agent)

qa_flow.add_edge('start', 'qa_agent')
qa_flow.add_edge('qa_agent', 'end')
qa_flow.compile()

from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

class DummyMetric(MetricBase):
    move = MInput(str, "answer")
    
    def score(self, label, move):
        return 0.5
