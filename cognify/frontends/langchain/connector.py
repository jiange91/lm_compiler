from langchain_core.runnables import Runnable, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from cognify.graph.base import StatePool
from cognify.llm import CogLM, InputVar, StructuredCogLM, OutputFormat, OutputLabel
from cognify.llm.model import LMConfig
import uuid
from litellm import ModelResponse
from typing import Any, List, Dict
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import warnings

APICompatibleMessage = Dict[str, str] # {"role": "...", "content": "..."}

langchain_message_role_to_api_message_role = {
  "system": "system",
  "human": "user",
  "ai": "assistant"
}

@dataclass
class LangchainOutput:
  content: str

DEFAULT_SYSTEM_PROMPT = "You are an intelligent assistant."
UNRECOGNIZED_PARAMS = ["model_name", "_type"]

class RunnableCogLM(Runnable):
  def __init__(self, runnable: Runnable = None, name: str = None):
    self.chat_prompt_template: ChatPromptTemplate = None
    self.cog_lm: CogLM = self.cognify_runnable(runnable, name)
  
  """
  Connector currently supports the following units to construct a `CogLM`:
  - BaseChatPromptTemplate | BaseChatModel
  - BaseChatPromptTemplate | BaseChatModel | BaseOutputParser
  These indepedent units should be split out of more complex chains.
  """
  def cognify_runnable(self, runnable: Runnable = None, name: str = None) -> CogLM:
    if not runnable:
      return None

    # parse runnable
    assert isinstance(runnable.first, ChatPromptTemplate), f"First runnable in a sequence must be a `ChatPromptTemplate` instead got {type(runnable.first)}"
    self.chat_prompt_template: ChatPromptTemplate = runnable.first
    output_parser = None

    if runnable.middle is None or len(runnable.middle) == 0:
      assert isinstance(runnable.last, BaseChatModel), f"Last runnable in a sequence with no middle must be a `BaseChatModel`, instead got {type(runnable.last)}"
      chat_model: BaseChatModel = runnable.last
    elif len(runnable.middle) == 1:
      assert isinstance(runnable.middle[0], BaseChatModel), f"Middle runnable must be a `BaseChatModel`, instead got {type(runnable.middle[0])}"
      chat_model: BaseChatModel = runnable.middle[0]

      assert isinstance(runnable.last, BaseOutputParser), f"Last runnable in a sequence with a middle `BaseChatModel` must be a `BaseOutputParser`, instead got {type(runnable.last)}"
      output_parser: BaseOutputParser = runnable.last
    else:
      raise NotImplementedError(f"Only one middle runnable is supported at this time, instead got {runnable.middle}")
    
    # initialize cog lm
    agent_name = runnable.name or name or str(uuid.uuid4())

    # system prompt
    if isinstance(self.chat_prompt_template.messages[0], SystemMessagePromptTemplate):
      system_message_prompt_template: SystemMessagePromptTemplate = self.chat_prompt_template.messages[0]
      if system_message_prompt_template.prompt.input_variables:
        raise NotImplementedError("Input variables are not supported in the system prompt. Best practices suggest placing these in")
      system_prompt_content: str = system_message_prompt_template.prompt.template
    else:
      warnings.warn("First message in a `ChatPromptTemplate` should be a `SystemMessagePromptTemplate`. Resorting to default system prompt", UserWarning)
      system_prompt_content: str = DEFAULT_SYSTEM_PROMPT

    # input variables (ignore partial variables)
    input_vars: List[InputVar] = [InputVar(name=name) for name in self.chat_prompt_template.input_variables]
    
    # lm config
    full_kwargs = chat_model._get_invocation_params()

    # remove unrecognized params 
    for param in UNRECOGNIZED_PARAMS:
      full_kwargs.pop(param, None)

    lm_config = LMConfig(model=full_kwargs.pop('model'), kwargs=full_kwargs)

    if output_parser is not None:
      output_format = OutputFormat(schema=output_parser.OutputType,
                                  should_hint_format_in_prompt=True,
                                  custom_output_format_instructions=output_parser.get_format_instructions())
      return StructuredCogLM(agent_name=agent_name,
                            system_prompt=system_prompt_content,
                            input_variables=input_vars,
                            output_format=output_format,
                            lm_config=lm_config)
    else:
      return CogLM(agent_name=agent_name,
                  system_prompt=system_prompt_content,
                  input_variables=input_vars,
                  output=OutputLabel(name="response"),
                  lm_config=lm_config)
  
  def invoke(self, input: Dict) -> Any:
    assert self.cog_lm, "CogLM must be initialized before invoking"

    messages = None
    if self.chat_prompt_template:
      chat_prompt_value: ChatPromptValue = self.chat_prompt_template.invoke(input)
      messages: List[APICompatibleMessage] = []
      for message in chat_prompt_value.messages:
        if message.type in langchain_message_role_to_api_message_role:
          messages.append({"role": langchain_message_role_to_api_message_role[message.type], "content": message.content})
        else:
          raise NotImplementedError(f"Message type {type(message)} is not supported, must be one of `SystemMessage`, `HumanMessage`, or `AIMessage`")
      
    result = self.cog_lm(messages, input) # kwargs have already been set when initializing cog_lm
    if isinstance(self.cog_lm, StructuredCogLM):
      return result
    else:
      return AIMessage(result)
  
def as_runnable(cog_lm: CogLM):
  runnable_cog_lm = RunnableCogLM(runnable=None, name=cog_lm.name)
  runnable_cog_lm.cog_lm = cog_lm
  return RunnableLambda(runnable_cog_lm.invoke)