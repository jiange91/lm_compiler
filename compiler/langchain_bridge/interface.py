from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_fireworks import ChatFireworks
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, BaseMessage, merge_message_runs, AIMessage
from langchain_aws import BedrockLLM
import threading

import traceback
from .utils import var_2_str

import logging
from typing import Tuple, Union, Callable, Type, Any, Dict, List
import types
from copy import deepcopy
import json

from compiler.IR.base import StatePool
from compiler.IR.llm import LLMPredictor, LMConfig, LMSemantic, Demonstration
from compiler.llm.schema_parser import get_pydantic_format_instruction as get_format_instruction
from compiler.llm.schema_parser import pydantic_model_repr
from compiler.langchain_bridge.utils import var_2_str
import copy
import os

logger = logging.getLogger(__name__)

    
from langchain_core.runnables import RunnableLambda

def inspect_with_msg(msg: str):
    def inspect_input(inputs, **kwargs):
        # print(msg, flush=True)
        # if isinstance(inputs, BaseMessage):
        #     print(var_2_str([inputs]), flush=True)
        # else:
        #     print(var_2_str(inputs), flush=True)
        return inputs
    return inspect_input

def get_inspect_runnable(msg: str = ""):
    return RunnableLambda(inspect_with_msg(msg))

class LLMTracker(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        # NOTE: tracker will not reset the cache
        # remember to clear this if needed
        self.llm_gen_meta_cache = []
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        pass
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        meta = {}
        
        usage = response.llm_output['token_usage']
        # print(usage)
        meta['completion_tokens'] = usage['completion_tokens']
        meta['prompt_tokens'] = usage['prompt_tokens']
        if 'completion_tokens_details' in usage:
            meta['reasoning_tokens'] = usage['completion_tokens_details']['reasoning_tokens']
        if 'prompt_tokens_details' in usage:
            meta['prompt_cached_tokens'] = usage['prompt_tokens_details']['cached_tokens']
            
        meta['model'] = response.llm_output['model_name']
        meta['response'] = response.generations[0][0].text
        self.llm_gen_meta_cache.append(deepcopy(meta))

class LangChainSemantic(LMSemantic):
    """Please do not set your format instructioins in the prompt
    just specify the output format with a pydantic model
    
    When applying optimization we need to control how to format your output schema bc 
    we might interleave the reasoning process with the output generation
    """
    def __init__(
        self,
        system_prompt: str,
        inputs: Union[str, list[str]],
        output_format: Union[Type[BaseModel], str],
        need_output_type_hint: bool = True,
        output_format_instructions: str = None,
        img_input_idices: list[int] = None,
        enable_memory: bool = False,
        input_key_in_mem: str = None,
        following_messages: list[MessageLikeRepresentation] = [],
        demos: list[Demonstration] = [],
    ):
        self.system_prompt = system_prompt
        # NOTE: this is short-term memory, only history wihtin a workflow execution is recorded
        assert not enable_memory, "memory flag experimental, use following_messages to manage explicitly"
        self.enable_memory = enable_memory
        if enable_memory:
            assert input_key_in_mem is not None, "Must provide input_key_in_mem if enable_memory is True"
        self.input_key_in_mem = input_key_in_mem
        
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        assert len(self.inputs) > 0, "At least one input is required"
        
        self.img_input_idices = img_input_idices
        self.img_input_names = set([self.inputs[i] for i in self.img_input_idices]) if self.img_input_idices else set()
        
        self.need_output_type_hint = need_output_type_hint
        self.output_format_instructions = output_format_instructions
        
        if isinstance(output_format, str):
            self.output_format = None
            self.parser = None
            self.outputs = [output_format]
            self.output_type_hint = None
        else:
            self.output_format = output_format
            self.parser = JsonOutputParser(pydantic_object=output_format)
            # NOTE: output name is inferred from the output format, use class name
            self.outputs = [self.output_format.__name__]
            if self.need_output_type_hint:
                self.output_type_hint = get_format_instruction(output_format)
        
        self.message_template_predefined = len(following_messages) > 0
        self.following_messages = following_messages
        self._chat_prompt_template: ChatPromptTemplate = None
        self.usr_demos = demos
        self.compiler_demos = []
    
    @property
    def demos(self) -> list[Demonstration]:
        return self.get_demos()
    
    def get_demos(self) -> list[Demonstration]:
        return self.usr_demos + self.compiler_demos
    
    def set_demos(self, demos: list[Demonstration]):
        self.compiler_demos = demos
    
    @property
    def chat_prompt_template(self):
        return self._chat_prompt_template

    @chat_prompt_template.setter
    def chat_prompt_template(self, value):
        self._chat_prompt_template = value
    
    def add_demos_to_prompt(self):
        input_fields = []
        for i, input in enumerate(self.inputs):
            if not self.img_input_idices or i not in self.img_input_idices:
                input_fields.append(f"{input}:\n${{{input}}}")
        
        example_format = "\n\n".join(input_fields) + \
                        "\n\nrationale:\nOptional(${reasoning})" + \
                        f"\n\n{self.outputs[0]}:\n${{{self.outputs[0]}}}"
        
        demo_strs = [
            f"Let me show you some examples following the format:\n\n{example_format}\n\n---\n\n"
        ]
        for i, demo in enumerate(self.demos):
            # add normal text inputs
            input_str = []
            for key, value in demo.inputs.items():
                if key not in self.img_input_names:
                    input_str.append(f"{key}:\n{value}")
            text_input_str = '\n\n'.join(input_str)
            demo_strs.append(f"{text_input_str}")
            for key, value in demo.inputs.items():
                if key in self.img_input_names:
                    self.chat_prompt_template.append(
                        HumanMessage(f"\n\n{key}:\n")
                    )
                    self.chat_prompt_template.append(
                        HumanMessage(
                            [{
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{value}"
                                }
                            }]
                        )
                    )
            if demo.reasoning:
                demo_strs.append(f"\n\nrationale:\n{demo.reasoning}")
            else:
                demo_strs.append("\n\nrationale:\nnot available")
                
            # add output, only consider text now
            demo_strs.append(
                f"\n\n{self.outputs[0]}:\n{demo.output}\n\n---\n\n"
            )
        demo_message = HumanMessage("".join(demo_strs))
        self.chat_prompt_template.append(demo_message)
        
    def build_prompt_template(self, strict_output: bool = True):
        # setup message list
        if not self.message_template_predefined:
            user_messages = []
            input_names = ", ".join(f"`{input}`" for input in self.inputs)
            strictly = " only" if strict_output else ""
            user_messages.append(
                {
                    "type": "text",
                    "text": f"Given {input_names}, please{strictly} provide `{self.outputs[0]}` in your response."
                }
            )
            if self.img_input_idices is not None:
                for img_idx in self.img_input_idices:
                    user_messages.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{{{self.inputs[img_idx]}}}"
                            }
                        }
                    )
            input_fields = []
            for i, input in enumerate(self.inputs):
                if not self.img_input_idices or i not in self.img_input_idices:
                    input_fields.append(f"{input}:\n{{{input}}}")
            real_usr_query = "\n".join(input_fields)
            user_messages.append(
                {
                    "type": "text",
                    "text": real_usr_query, 
                }
            )
            self.usr_prompt_template = HumanMessagePromptTemplate.from_template(template=user_messages) # temporarily remove image support
            self.following_messages = [self.usr_prompt_template]
        # setup prompt template
        if self.enable_memory:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    MessagesPlaceholder(variable_name="compiler_chat_history"),
                ]
            )
        else:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                ]
            )
        
        # add few-shot examples
        if self.demos:
            self.add_demos_to_prompt()
        
        # add user prompt
        self.chat_prompt_template.extend(self.following_messages)
        # add all output format instructions
        ospec = self.get_output_format_spec()
        if ospec:
            self.chat_prompt_template.append(ospec)
    
    def get_output_format_spec(self) -> HumanMessage | None:
        msgs = []
        if self.output_type_hint:
            msgs.append(HumanMessage(self.output_type_hint))
        if self.output_format_instructions:
            msgs.append(HumanMessage(self.output_format_instructions))
        if len(msgs) == 0:
            return None
        return merge_message_runs(msgs)[0]
    
    def get_output_spec(self) -> Tuple[bool, str | None]:
        return self.need_output_type_hint, self.output_format_instructions
    
    def prompt_fully_manageable(self) -> bool:
        return not self.message_template_predefined

    def get_agent_role(self) -> str:
        return self.system_prompt

    def get_agent_inputs(self) -> list[str]:
        return self.inputs

    def get_agent_outputs(self) -> list[str]:
        return self.outputs

    def get_output_schema(self) -> type[BaseModel]:
        return self.output_format

    def get_high_level_info(self) -> str:
        dict = {
            "agent_prompt": self.system_prompt,
            "input_names": self.inputs,
            "output_name": self.outputs[0],
        }
        return json.dumps(dict, indent=4)

    # TODO: user provided output format instruction should be considered in decomposition of the task
    def get_formatted_info(self) -> str:
        if self.output_format is not None:
            output_schemas = self.output_format.model_json_schema()
        else:
            output_schemas = self.outputs[0]
        dict = {
            "agent_prompt": self.system_prompt,
            "input_variables": self.inputs,
            "output_schema": output_schemas
        }
        return json.dumps(dict, indent=4)

    def get_img_input_names(self) -> List[str]:
        if self.img_input_idices is None:
            return []
        else:
            return [self.inputs[i] for i in self.img_input_idices]


class LangChainLM(LLMPredictor):
    def __init__(self, name, semantic: LangChainSemantic, lm=None, lm_config=None, **kwargs) -> None:
        """
        """
        self.chat_history = ChatMessageHistory()
        self.semantic: LangChainSemantic = semantic # mainly for type hint
        # default model can be overwritten by set_lm
        self.lm = lm
        self.lm_config = lm_config
        self.reasoning = None
        self._tracker = LLMTracker()
        super().__init__(name, semantic, self.lm, self.lm_config, **kwargs)
    
    def get_invoke_routine(self):
        if self.reasoning is not None:
            return self.reasoning.get_invoke_routine(self)
        self.semantic.build_prompt_template() # will rebuild the prompt template for each re-compile
        inputs_str = ', '.join(self.semantic.inputs)
        invoke_arg_dict_str = '{' + ', '.join(
                [f'"{input}": {input}' for input in self.semantic.inputs] 
            ) + '}'
        #NOTE: use imperative merge at runtime bc message placeholder cannot be merged statically
        routine = self.semantic.chat_prompt_template | get_inspect_runnable(f'-- {self.name} input --') | self.lm | get_inspect_runnable(f'-- {self.name} output --')
        if self.semantic.enable_memory:
            routine = RunnableWithMessageHistory(
                runnable=routine,
                get_session_history=lambda: self.chat_history,
                input_messages_key=self.semantic.input_key_in_mem,
                history_messages_key="compiler_chat_history",
            )
        if self.semantic.output_format:
            result_str = f'{{"{self.semantic.outputs[0]}": result}}'
            routine = routine | self.semantic.parser
            langchain_kernel_template = f"""
def langchain_lm_kernel({inputs_str}):
    try:
        # print("langchain lm output parse")
        result = routine.invoke({invoke_arg_dict_str})
        result = output_format.model_validate(result)
        # print(result)
        return {result_str}
    except Exception as e:
        print(e)
        print("ERR IN langchain_lm output parse")
        raise
"""
        else:
            result_str = f'{{"{self.semantic.outputs[0]}": result.content}}'
            langchain_kernel_template = f"""
def langchain_lm_kernel({inputs_str}):
    try:
        # print("langchain lm no output parse")
        result = routine.invoke({invoke_arg_dict_str})
        return {result_str}
    except Exception as e:
        print(e)
        print("ERR IN langchain_lm no output parse")
        raise
"""
        self.kernel_str = langchain_kernel_template
        func_obj = compile(langchain_kernel_template, '<string>', 'exec')
        local_name_space = {}
        exec(func_obj, 
             {
                'routine': routine, 
                'output_format': self.semantic.output_format,
                'chat_template': self.semantic.chat_prompt_template,
            }, local_name_space)
        return local_name_space['langchain_lm_kernel']
    

    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        # self.lm = ChatOpenAI(
        #     model='gpt-4o-mini',
        #     **self.lm_config.kwargs,
        #     api_key=os.environ['OPENAI_API_KEY'],
        #     callbacks=[LLMTracker(self)]
        # )
        # return
        if self.lm_config.provider == 'openai':
            self.lm = ChatOpenAI(
                model=self.lm_config.model,
                **self.lm_config.kwargs,
                api_key=os.environ['OPENAI_API_KEY'],
                callbacks=[self._tracker]
            )
        elif self.lm_config.provider == 'together':
            self.lm = ChatTogether(
                model=self.lm_config.model,
                **self.lm_config.kwargs,
                api_key=os.environ['TOGETHER_API_KEY'],
                callbacks=[self._tracker]
            )
        elif self.lm_config.provider == 'fireworks':
            self.lm = ChatFireworks(
                model=self.lm_config.model,
                **self.lm_config.kwargs,
                api_key=os.environ['FIREWORKS_API_KEY'],
                callbacks=[self._tracker],
            )
        elif self.lm_config.provider == 'local':
            base_url = self.lm_config.kwargs.get('openai_api_base', None)
            if base_url is None:
                raise ValueError("Local provider requires openai_api_base")
            self.lm = ChatOpenAI(
                model=self.lm_config.model,
                **self.lm_config.kwargs,
                api_key="DUMMY",
                callbacks=[self._tracker]
            )
        else:
            raise ValueError(f"Provider {self.lm_config.provider} not supported")
        return

    def get_lm_history(self):
        """get the history of the last generation
        
        remember to clear the history after getting it
        """
        hist_cpy = deepcopy(self._tracker.llm_gen_meta_cache)
        self._tracker.llm_gen_meta_cache.clear()
        return hist_cpy

    def get_step_as_example(self) -> Demonstration:
        if len(self.step_info) == 0:
            return None
        
        step = self.step_info[-1]
        inputs = step['inputs']
        output = step['output']
        if not isinstance(output, str):
            raise ValueError(f"Output must be a string, got {output}")
        inputs_dict = {}
        for key, value in inputs.items():
            inputs_dict[key] = var_2_str(value)
        demo = Demonstration(
            inputs=inputs_dict,
            output=output,
            reasoning=step.get('rationale', None)
        )
        return demo

    def custom_reset(self):
        self._tracker.llm_gen_meta_cache.clear()
        self.chat_history.clear()
    
    
    def as_runnable(self):
        def invoke(input: dict):
            try:
                statep = StatePool()
                statep.init(input)
                self.invoke(statep)
                result = statep.news(self.semantic.outputs[0])
                if self.semantic.output_format:
                    return result
                else:
                    return AIMessage(result)
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}")
                traceback.print_exc()
                raise
        return RunnableLambda(invoke)