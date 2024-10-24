from dataclasses import dataclass
from typing import List, Dict, Literal, Optional, override
from compiler.llm.prompt import InputVariable, CompletionMessage, Demonstration, Content, TextContent
from compiler.llm.output import OutputFormat, get_format_hint
import litellm
from litellm import completion, get_supported_openai_params
from pydantic import BaseModel

APICompatibleMessage = Dict[str, str]

@dataclass
class LMConfig:
  model: str # see https://docs.litellm.ai/docs/providers 
  custom_llm_provider: Optional[str]
  kwargs: Optional[dict]
  relative_cost: float = 1.0  # used to rank models during model selection optimization step

class CogLM:
  def __init__(self, system_prompt: str, 
               input_variables: List[InputVariable], 
               model_config: LMConfig, 
               output_label: str = None):
    self.system_message: CompletionMessage = CompletionMessage(role="system", content=[TextContent(text=system_prompt)])
    self.input_variables: List[InputVariable] = input_variables
    self.model_config: LMConfig = model_config
    self.output_label: str = output_label
    self.demo_messages: List[CompletionMessage] = []

  def add_demos(self, demos: List[Demonstration], demo_prompt_string: str = None):
    if not demos:
      raise Exception("No demonstrations provided")

    input_variable_names = [variable.name for variable in self.input_variables]
    demo_prompt_string = demo_prompt_string or "Let me show you some examples following the format" # customizable
    demos_content: List[Content] = []

    demos_content.append(TextContent(text=f"""{demo_prompt_string}:\n\n{self._get_example_format()}--\n\n"""))
    for demo in demos:
      # validate demonstration
      demo_variable_names = [filled.input_variable.name for filled in demo.filled_input_variables]
      if set(demo_variable_names) != set(input_variable_names):
        raise ValueError(f'Demonstration variables {demo_variable_names} do not match input variables {input_variable_names}')
      else:
        demos_content.extend(demo.to_content())
    self.demo_messages.append(CompletionMessage(role="user", content=demos_content))

  def _get_example_format(self):
    input_fields = []
    for variable in self.input_variables:
      input_fields.append(f"{variable.name}:\n${{{variable.name}}}") # e.g. "question: ${question}"

    return "\n\n".join(input_fields) + \
      "\n\nrationale:\nOptional(${reasoning})" + \
      f"\n\n{self._get_output_label()}:\n${{{self._get_output_label()}}}"
  
  def _get_output_label(self):
    return self.output_label or "response"

  def _get_api_compatible_messages(self, messages: List[APICompatibleMessage]) -> List[APICompatibleMessage]:
    return self.system_message.to_api() + messages + [demo_message.to_api() for demo_message in self.demo_messages]

  def forward(self, messages: List[APICompatibleMessage]):
    return completion(self.model_config.model, 
                      self._get_api_compatible_messages(messages),
                      custom_llm_provider=self.model_config.custom_llm_provider, 
                      **self.model_config.kwargs)

class StructuredCogLM(CogLM):
  def __init__(self, system_prompt: str, 
               input_variables: List[InputVariable], 
               model_config: LMConfig, 
               output_format: OutputFormat):
    self.output_format: OutputFormat = output_format
    super(CogLM, self).__init__(system_prompt, input_variables, model_config, self._get_output_label())

  @override
  def _get_output_label(self):
    return self.output_format.schema.__name__
  
  @override
  def _get_api_compatible_messages(self, messages: List[APICompatibleMessage]) -> List[APICompatibleMessage]:
    return super(CogLM, self)._get_api_compatible_messages(messages) + [self.output_format.get_output_instruction_message().to_api()]

  @override
  def forward(self, messages: List[APICompatibleMessage]):
    litellm.enable_json_schema_validation = True
    params = get_supported_openai_params(model=self.model_config.model, 
                                         custom_llm_provider=self.model_config.custom_llm_provider)
    if "response_format" not in params:
      raise ValueError(f"Model {self.model_config.model} on provider {self.model_config.custom_llm_provider} does not support structured output") 
    else:
      self.messages.append(self.output_format.get_output_instruction_message())
      return completion(self.model_config.model, 
                        self._get_api_compatible_messages(messages),
                        custom_llm_provider=self.model_config.custom_llm_provider,
                        response_format=self.output_format,
                        **self.model_config.kwargs)