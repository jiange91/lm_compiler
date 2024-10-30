from langchain_core.runnables import Runnable, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai.chat_models.base import BaseChatOpenAI
from compiler.llm import CogLM, InputVar, StructuredCogLM, OutputFormat
from compiler.llm.model import LMConfig
import uuid
from litellm import ModelResponse
from typing import Any, List, Dict

APICompatibleMessage = Dict[str, str] # {"role": "...", "content": "..."}

langchain_message_role_to_api_message_role = {
  "system": "system",
  "human": "user",
  "ai": "assistant"
}

class RunnableCogLM(Runnable):
  def __init__(self, runnable: Runnable = None, name: str = None):
    self.chat_prompt_template: ChatPromptTemplate = None
    self.cog_lm: CogLM = self.cognify_runnable(runnable, name)
  
  """
  Connector currently supports the following units to construct a `CogLM`:
  - BaseChatPromptTemplate | BaseChatOpenAI
  - BaseChatPromptTemplate | BaseChatOpenAI | BaseOutputParser
  These indepedent units should be split out of more complex chains.
  """
  def cognify_runnable(self, runnable: Runnable = None, name: str = None) -> CogLM:
    if not runnable:
      return None

    # parse runnable
    assert isinstance(runnable, RunnableSequence), "Runnable must be a `RunnableSequence`"
    assert isinstance(runnable.first, ChatPromptTemplate), "First runnable in a sequence must be a `ChatPromptTemplate`"
    self.chat_prompt_template: ChatPromptTemplate = runnable.first
    output_parser = None

    if runnable.middle is None:
      assert isinstance(runnable.last, BaseChatOpenAI), "Last runnable in a sequence with no middle must be a `BaseChatOpenAI`"
      chat_model: BaseChatOpenAI = runnable.last
    elif len(runnable.middle) == 1:
      assert isinstance(runnable.middle[0], BaseChatOpenAI), "Middle runnable must be a `BaseChatOpenAI`"
      chat_model: BaseChatOpenAI = runnable.middle[0]

      assert isinstance(runnable.last, BaseOutputParser), "Last runnable in a sequence with a middle `BaseChatOpenAI` must be a `BaseOutputParser`"
      output_parser: BaseOutputParser = runnable.last
    else:
      raise NotImplementedError("Only one middle runnable is supported at this time")
    
    # initialize cog lm
    agent_name = runnable.name or name or str(uuid.uuid4())

    # system prompt
    assert isinstance(self.chat_prompt_template.messages[0], SystemMessagePromptTemplate), "First message in a `ChatPromptTemplate` must be a `SystemMessagePromptTemplate`"
    system_message_prompt_template: SystemMessagePromptTemplate = self.chat_prompt_template.messages[0]
    if system_message_prompt_template.prompt.input_variables:
      raise NotImplementedError("Input variables are not supported in the system prompt. Best practices suggest placing these in")
    system_prompt_content: str = system_message_prompt_template.prompt.template

    # input variables (ignore partial variables)
    input_vars: List[InputVar] = [InputVar(name=name) for name in self.chat_prompt_template.input_variables]
    
    # lm config
    full_kwargs = chat_model._get_invocation_params()
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
                  lm_config=lm_config)

  def invoke(self, input: Dict) -> Any:
    assert self.cog_lm, "CogLM must be initialized before invoking"

    chat_prompt_value: ChatPromptValue = self.chat_prompt_template.invoke(input)
    messages = self._get_api_compatible_messages(chat_prompt_value)
    inputs: Dict[InputVar, str] = {InputVar(name=k): v for k,v in input.items()}
    result: ModelResponse = self.cog_lm.forward(messages, inputs) # kwargs have already been set when initializing cog_lm
    if isinstance(self.cog_lm, StructuredCogLM):
      return self.cog_lm.output_format.schema.model_validate_json(result)
    else:
      return result.choices[0].message.content
    
  def _get_api_compatible_messages(chat_prompt_value: ChatPromptValue) -> List[APICompatibleMessage]:
    api_comptaible_messages: List[APICompatibleMessage] = []
    for message in chat_prompt_value.messages:
      if message.type in langchain_message_role_to_api_message_role:
        api_comptaible_messages.append({"role": langchain_message_role_to_api_message_role[message.type], "content": message.content})
      else:
        raise NotImplementedError(f"Message type {type(message)} is not supported, must be one of `SystemMessage`, `HumanMessage`, or `AIMessage`")
    return api_comptaible_messages
  

def as_runnable(cog_lm: CogLM) -> RunnableCogLM:
  runnable_cog_lm = RunnableCogLM(runnable=None, name=cog_lm.agent_name)
  runnable_cog_lm.cog_lm = cog_lm
  return runnable_cog_lm