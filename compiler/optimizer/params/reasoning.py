from abc import ABC, ABCMeta
from typing import List, Optional
import traceback
from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase, IdentityOption
from compiler.llm import CogLM, StructuredCogLM, StepInfo, InputVar, OutputFormat, OutputLabel
from compiler.llm.model import APICompatibleMessage
from litellm import ModelResponse, completion
import copy

import logging

logger = logging.getLogger(__name__)

class LMReasoning(ParamBase):
    level = ParamLevel.NODE
    
    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = data['name'], data['module_name'], data['default_option'], data['options']
        options = [ReasonThenFormat.registry[dat['type']].from_dict(dat) for name, dat in options.items()]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )

class ReasoningOptionMeta(ABCMeta):
    registry: dict[str, type] = {'IdentityOption': IdentityOption}
    
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls

class ReasonThenFormat(OptionBase, metaclass=ReasoningOptionMeta):
    
    @classmethod
    def direct_apply(cls, lm_module: CogLM):
        reasoning = cls()
        reasoning.apply(lm_module)
        return reasoning

    def reasoning_step(
        self, 
        model: str,
        chat_messages: List[APICompatibleMessage],
        model_kwargs: dict
    ) -> List[ModelResponse]:
        """Produce reasoning steps for the given chat prompt messages
        """
        raise NotImplementedError

    def get_all_model_responses(self):
        return self.model_responses

    def aggregate_reasoning_steps(self, responses: List[ModelResponse]) -> str:
        agg_messages = []
        for response in responses:
            agg_messages.append(f"\n: {response.choices[0].message.content}")
        return "\n".join(agg_messages)

    def forward(self, lm_module: CogLM, messages: List[APICompatibleMessage], model_kwargs: Optional[dict] = None) -> List[ModelResponse]:
        """
        If the orignal output has certain format, applying additional reasoning steps will break down
        it into two phases, first one allows free generation along with reasoning steps, and the second
        one will the formatting step
        """
        self.model_responses = []

        if not model_kwargs:
            assert lm_module.lm_config, "Model kwargs must be provided if LM config is not set at initialization"
        full_kwargs = model_kwargs or lm_module.lm_config.get_model_kwargs()
        model: str = full_kwargs.pop("model")
        responses: List[ModelResponse] = []

        messages.append({"role": "user", "content": "Don't give your final response to the instruction directly. We can start with some reasoning first.\n"})
        reasoning_step_responses: List[ModelResponse] = self.reasoning_step(copy.deepcopy(messages))
        responses.extend(self.get_all_model_responses())
        rationale = self.aggregate_reasoning_steps(reasoning_step_responses)
        lm_module.rationale = rationale

        messages.append({"role": "assistant", "content": rationale})
        if lm_module.contains_custom_format_instructions():
            messages.append({"role": "user", "content": f"Now please only give {lm_module.get_output_label_name()} according to the following instructions:\n{lm_module.get_custom_format_instructions_if_any()}"})
        else:
            messages.append({"role": "user", "content": f"Based on all this information, please only provide {lm_module.get_output_label_name()}."})
              
        full_messages = [lm_module.system_message.to_api()] + messages
        if isinstance(lm_module, StructuredCogLM):
            response = completion(model, 
                                full_messages, 
                                response_format=lm_module.output_format.schema, 
                                **full_kwargs)
            responses.append(response)
        else:
            response = completion(model, 
                                full_messages,
                                **full_kwargs)
            responses.append(response)
        return responses
    
    def apply(self, lm_module: CogLM):
        lm_module.reasoning = self
        return lm_module

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

class ZeroShotCoT(ReasonThenFormat):
    def __init__(self):
        super().__init__("ZeroShotCoT")
    
    def _get_cost_indicator(self):
        return 4.0

    def reasoning_step(
        self, 
        model: str,
        chat_messages: List[APICompatibleMessage],
        model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append({"role": "user", "content": "Let's solve this problem step by step before giving the final response\n"})
        response = completion(model, chat_messages, **model_kwargs)
        return [response]
        

class PlanBefore(ReasonThenFormat):
    def __init__(self):
        super().__init__("PlanBefore")
    
    def _get_cost_indicator(self):
        return 2.5
    
    def reasoning_step(
        self, 
        model: str,
        chat_messages: List[APICompatibleMessage],
        model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append({"role": "user", "content": "Let's first plan necessary steps to approach this problem before giving the final response\n"})
        response = completion(model, chat_messages, **model_kwargs)
        return [response]
