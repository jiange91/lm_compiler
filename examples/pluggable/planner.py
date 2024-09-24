from compiler.utils import load_api_key, get_bill
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.output_parsers import StrOutputParser
import operator
from typing import Annotated, List, Tuple, TypedDict
from copy import deepcopy
from typing import List
from compiler.langchain_bridge.interface import get_inspect_runnable

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
    base
)
from langchain_openai.chat_models import ChatOpenAI

#==============================================================================
# Planner
#==============================================================================

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

plan_system = "For the given task, come up with a simple step by step plan. Please provide a plan with at most 3 steps."

plan_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=plan_system),
        HumanMessagePromptTemplate.from_template("Please fufill the following task: {task}")
    ]
)

planner = plan_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(Plan)

#==============================================================================
# With Annotation
#==============================================================================
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.optimizer import registry

semantic = LangChainSemantic(
    system_prompt=plan_system,
    inputs=["task"],
    output_format=Plan,
    output_format_instructions=""
)

planner_anno = LangChainLM('planner', semantic, opt_register=True)
planner_anno.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}
runnable_planner_anno = planner_anno.as_runnable()
