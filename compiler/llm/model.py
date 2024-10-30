from dataclasses import dataclass
from typing import List, Dict, Optional, override
from compiler.llm.prompt import InputVar, CompletionMessage, Demonstration, Content, TextContent, ImageContent, FilledInputVar, get_image_content_from_upload
from compiler.llm.output import OutputLabel, OutputFormat
import litellm
from litellm import completion, get_supported_openai_params, ModelResponse
from pydantic import BaseModel
import json
from litellm import Usage
from openai.types import CompletionUsage
from compiler.llm.response import ResponseMetadata, aggregate_usages, StepInfo
from compiler.IR.base import Module
import copy
import threading
import logging

logger = logging.getLogger(__name__)
APICompatibleMessage = Dict[str, str] # {"role": "...", "content": "..."}
_thread_local_chain = threading.local()

def _local_forward(_local_lm: 'CogLM', messages: List[APICompatibleMessage], inputs: Dict[str, str], model_kwargs: Optional[dict] = None):
  if _local_lm.reasoning:
    responses: List[ModelResponse] = _local_lm.reasoning.forward(_local_lm, messages, model_kwargs)
    _local_lm.response_metadata_history.extend([ResponseMetadata(model=response.model, 
                                                                 cost=response._hidden_params["response_cost"], 
                                                                 usage=response.usage) for response in responses])
  else:
    response: ModelResponse = _local_lm._forward(messages, inputs, model_kwargs)
    _local_lm.response_metadata_history.append(ResponseMetadata(model=response.model, 
                                                                cost=response._hidden_params["response_cost"], 
                                                                usage=response.usage))
  step_info = StepInfo(filled_inputs_dict=inputs, 
                       output=response.choices[0].message.content,
                       rationale=_local_lm.rationale)
  _local_lm.steps.append(step_info)
  _local_lm.rationale = None
  
  return response.choices[0].message.content

@dataclass
class LMConfig:
  model: str # see https://docs.litellm.ai/docs/providers 
  kwargs: Optional[dict]
  custom_llm_provider: Optional[str] = None
  cost_indicator: float = 1.0  # used to rank models during model selection optimization step

  def to_dict(self):
    return self.__dict__
  
  @classmethod
  def from_dict(cls, data):
    obj = cls.__new__(cls)
    obj.__dict__.update(data)
    return obj

  def get_model_kwargs(self) -> dict:
    full_kwargs_dict = self.kwargs or {}
    full_kwargs_dict["model"] = self.model
    if self.custom_llm_provider:
      full_kwargs_dict["custom_llm_provider"] = self.custom_llm_provider
    return full_kwargs_dict
  
  def update(self, other: 'LMConfig'):
    self.model = other.model
    self.custom_llm_provider = other.custom_llm_provider
    self.kwargs.update(other.kwargs)
    self.cost_indicator = other.cost_indicator

class CogLM(Module):
  def __init__(self, agent_name: str,
               system_prompt: str, 
               input_variables: List[InputVar], 
               output: Optional[OutputLabel] = None,
               lm_config: Optional[LMConfig] = None,
               opt_register: bool = True):
    self._lock = threading.Lock()
    super().__init__(name=agent_name, kernel=None, opt_register=opt_register)

    self.system_message: CompletionMessage = CompletionMessage(role="system", content=[TextContent(text=system_prompt)])
    self.input_variables: List[InputVar] = input_variables
    self.output_label: Optional[OutputLabel] = output
    self.demo_messages: List[CompletionMessage] = []
    self.response_metadata_history: List[ResponseMetadata] = []
    self.steps: List[StepInfo] = []
    self.reasoning = None
    self.rationale = None

    # TODO: improve lm configuration handling between agents. currently just unique config for each agent
    self.lm_config = copy.deepcopy(lm_config)

    setattr(_thread_local_chain, agent_name, self)

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
      if k != '_lock':
        setattr(result, k, copy.deepcopy(v, memo))
      else:
        setattr(result, k, threading.Lock())
    return result
  
  @override
  def reset(self):
    super().reset()
    self.response_metadata_history = []
    self.steps = []
    self.lm_config = None

  def get_thread_local_chain(self):
    try:
      if not hasattr(_thread_local_chain, self.name):
        # NOTE: no need to set to local storage bc that's mainly used to detect if current context is in a new thread
        _self = copy.deepcopy(self)
        _self.reset()
      else:
        _self = getattr(_thread_local_chain, self.name)
      return _self
    except Exception as e:
      logger.info(f'Error in get_thread_local_chain: {e}')
      raise

  def get_high_level_info(self) -> str:
    dict = {
      "agent_prompt": self._get_system_prompt(),
      "input_names": self._get_input_names(),
      "output_name": self.get_output_label_name(),
    }
    return json.dumps(dict, indent=4)
  
  def get_formatted_info(self) -> str:
    dict = {
      "agent_prompt": self._get_system_prompt(),
      "input_variables": self._get_input_names(),
      "output_schema": self.get_output_label_name(),
    }
    return json.dumps(dict, indent=4)
  
  def get_last_step_as_demo(self) -> Optional[Demonstration]:
    if not self.steps:
      return None
    else:
      last_step: StepInfo = self.steps[-1]
      filled_input_dict: Dict[str, str] = {} # input name -> input value
      for input_variable in self.input_variables:
        input_value = last_step.filled_inputs_dict.get(input_variable.name, None)
        filled_input_dict[input_variable.name] = input_value
      return Demonstration(inputs=filled_input_dict, 
                           output=last_step.output, 
                           reasoning=last_step.rationale)

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

  def get_lm_response_metadata_history(self) -> List[ResponseMetadata]:
    return copy.deepcopy(self.response_metadata_history)

  def get_total_cost(self) -> float:
    return sum([response_metadata.cost for response_metadata in self.response_metadata_history])
  
  def get_current_token_usages(self) -> List[Usage]:
    return copy.deepcopy([response_cost_usage.usage for response_cost_usage in self.response_metadata_history])

  def get_aggregated_token_usage(self) -> CompletionUsage:
    return aggregate_usages(self.get_current_token_usages())

  def _get_example_format(self):
    input_fields = []
    for variable in self.input_variables:
      input_fields.append(f"{variable.name}:\n${{{variable.name}}}") # e.g. "question: ${question}"

    return "\n\n".join(input_fields) + \
      "\n\nrationale:\nOptional(${reasoning})" + \
      f"\n\n{self.get_output_label_name()}:\n${{{self.get_output_label_name()}}}"
  
  def get_output_label_name(self) -> str:
    return self.output_label.name or "response"

  def _get_input_names(self) -> List[str]:
    return [variable.name for variable in self.input_variables]

  def _get_system_prompt(self) -> str:
    return self.system_message.content[0].text
  
  def get_agent_role(self) -> str:
    return self._get_system_prompt()
  
  def contains_custom_format_instructions(self) -> bool:
    return self.output_label and self.output_label.custom_output_format_instructions

  def get_custom_format_instructions_if_any(self) -> Optional[str]:
    return self.output_label.custom_output_format_instructions if self.contains_custom_format_instructions() else None

  def _get_api_compatible_messages(self, messages: List[APICompatibleMessage]) -> List[APICompatibleMessage]:
    if messages[0]["role"] == "system":
      messages = messages[1:]
    
    api_compatible_messages = [self.system_message.to_api()] + messages
    api_compatible_messages.extend([demo_message.to_api() for demo_message in self.demo_messages])
    if self.contains_custom_format_instructions():
      api_compatible_messages.append({"role": "user", "content": self.output.custom_output_format_instructions})
    return api_compatible_messages

  def aggregate_thread_local_meta(self, _local_self: 'CogLM'):
    if self is _local_self:
      return
    with self._lock:
      self.steps.extend(_local_self.steps)
      self.response_metadata_history.extend(_local_self.response_metadata_history)

  def __call__(self, messages: List[APICompatibleMessage] = [], inputs: Dict[InputVar|str, str] = None, model_kwargs: Optional[dict] = None) -> ModelResponse:
    if inputs and isinstance(list(inputs.keys())[0], InputVar):
      # strip down the passed input
      inputs = {input_var.name: value for input_var, value in inputs.items()}
    return self.forward(messages, inputs, model_kwargs)

  def forward(self, messages: List[APICompatibleMessage] = [], inputs: Dict[str, str] = None, model_kwargs: Optional[dict] = None) -> ModelResponse:
    _self = self.get_thread_local_chain()
    result = _local_forward(_self, messages, inputs, model_kwargs)
    self.aggregate_thread_local_meta(_self)
    return result

  def _get_input_messages(self, inputs: Dict[str, str]) -> List[APICompatibleMessage]:
    assert set(inputs.keys()) == set([input.name for input in self.input_variables]), "Input variables do not match"

    input_names = ", ".join(f"`{name}`" for name in inputs.keys())
    messages = [CompletionMessage(role="user", 
                                  content=[TextContent(text=f"Given {input_names}, please strictly provide `{self.get_output_label_name()}`")])]
    
    input_fields = []
    for input_var in self.input_variables:
      if input_var.image_params:
        if input_var.image_params.is_image_upload:
          image_content = get_image_content_from_upload(inputs[input_var.name], input_var.image_params.file_type)
        else:
          image_content = ImageContent(image_url=inputs[input_var.name])
        messages.append(CompletionMessage(role="user", 
                                          content=[image_content]))
      else:
        input_fields.append(f"{input_var.name}: {inputs[input_var.name]}")
    messages.append(CompletionMessage(role="user", 
                                      content=[TextContent(text="\n".join(input_fields))]))
    return messages


  def _forward(self, messages: List[APICompatibleMessage] = [], inputs: Dict[str, str] = None, model_kwargs: Optional[dict] = None) -> ModelResponse:
    assert messages or inputs, "Either messages or inputs must be provided"
    if not messages:
      messages = self._get_input_messages(inputs)
    
    if not model_kwargs:
      assert self.lm_config, "Model kwargs must be provided if LM config is not set at initialization"
    
    full_kwargs = model_kwargs or self.lm_config.get_model_kwargs()
    model = full_kwargs.pop("model")
    response: ModelResponse = completion(model, 
                      self._get_api_compatible_messages(messages),
                      **full_kwargs)
    return response
  

class StructuredCogLM(CogLM):
  def __init__(self, agent_name: str, 
               system_prompt: str, 
               input_variables: List[InputVar],
               output_format: OutputFormat,
               lm_config: Optional[LMConfig] = None,
               opt_register: bool = True):
    self.output_format: OutputFormat = output_format
    super().__init__(agent_name, system_prompt, input_variables, lm_config=lm_config, opt_register=opt_register)

  @override
  def get_output_label_name(self):
    return self.output_format.schema.__name__
  
  @override
  def get_formatted_info(self) -> str:
    dict = {
      "agent_prompt": self._get_system_prompt(),
      "input_variables": self._get_input_names(),
      "output_schema": self.output_format.schema.model_json_schema()
    }
    return json.dumps(dict, indent=4)

  @override
  def contains_custom_format_instructions(self) -> bool:
    return self.output_format.custom_output_format_instructions is not None

  @override
  def get_custom_format_instructions_if_any(self) -> Optional[str]:
    return self.output_label.custom_output_format_instructions

  @override
  def _get_api_compatible_messages(self, messages: List[APICompatibleMessage]) -> List[APICompatibleMessage]:
    api_compatible_messages = super(CogLM, self)._get_api_compatible_messages(messages)
    api_compatible_messages.append(self.output_format.get_output_instruction_message().to_api())
    return api_compatible_messages

  @override
  def _forward(self, messages: List[APICompatibleMessage], inputs: Dict[str, str] = None, model_kwargs: Optional[dict] = None) -> ModelResponse:
    litellm.enable_json_schema_validation = True

    assert messages or inputs, "Either messages or inputs must be provided"
    if not messages:
      messages = self._get_input_messages(inputs)

    if not model_kwargs:
      assert self.lm_config, "Model kwargs must be provided if LM config is not set at initialization"

    full_kwargs = model_kwargs or self.lm_config.get_model_kwargs()
    params = get_supported_openai_params(model=full_kwargs["model"], 
                                         custom_llm_provider=full_kwargs["custom_llm_provider"])
    if "response_format" not in params:
      raise ValueError(f"Model {full_kwargs["model"]} on provider {full_kwargs["custom_llm_provider"]} does not support structured output") 
    else:
      model = full_kwargs.pop('model')
      messages.append(self.output_format.get_output_instruction_message())
      response: ModelResponse = completion(model, 
                        self._get_api_compatible_messages(messages),
                        response_format=self.output_format.schema,
                        **full_kwargs)
      return response