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

from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic

system = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

class DocumentFilter(BaseModel):
    decision: str = Field(description="yes or no")

doc_filter_semantic = LangChainSemantic(
    system_prompt=system,
    inputs=["question", "doc"],
    output_format='decision',
)

qa_agent = LangChainLM('doc_filter', doc_filter_semantic)
qa_agent.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}


from compiler.IR.program import Workflow, Input, Output, StatePool

qa_flow = Workflow('qa_flow')

qa_flow.add_module(Input('start'))
qa_flow.add_module(Output('end'))
qa_flow.add_module(qa_agent)

qa_flow.add_edge('start', 'doc_filter')
qa_flow.add_edge('doc_filter', 'end')
qa_flow.compile()

