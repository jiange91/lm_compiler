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
# Executor
#==============================================================================

exec_system = "Given a task and the corresponding pre-defined steps, please provide your response following the steps."
exec_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=exec_system),
        HumanMessagePromptTemplate.from_template("Task: {task}\n\nSteps:\n{steps}")
    ]
)

executor = exec_prompt | get_inspect_runnable() | ChatOpenAI(model="gpt-4o-mini", temperature=0.0) | StrOutputParser()

#==============================================================================
# With Annotation
#==============================================================================
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

semantic = LangChainSemantic(
    system_prompt=exec_system,
    inputs=["task", "steps"],
    output_format="response",
)

exec_anno = LangChainLM('executor', semantic, opt_register=True)
exec_anno.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}
runnable_exec_anno = exec_anno.as_runnable()