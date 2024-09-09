from langchain_openai import ChatOpenAI
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.prompts.chat import MessageLikeRepresentation
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage

import logging
from typing import Union, Callable, Type
import types
from copy import deepcopy
import json

from compiler.IR.llm import LLMPredictor, LMConfig, LMSemantic

logger = logging.getLogger(__name__)

def get_format_instruction(schema: BaseModel):
    example_json_schema = """
```json
{
    "title": "ComplexityList",
    "description": "complexity of all agents",
    "type": "object",
    "properties": {
        "es": {
            "title": "Es",
            "description": "list of complexity descriptions",
            "type": "array",
            "items": {
                "$ref": "#/definitions/ComplexityEstimation"
            }
        }
    },
    "required": [
        "es"
    ],
    "definitions": {
        "ComplexityEstimation": {
            "title": "ComplexityEstimation",
            "description": "complexity of each agent",
            "type": "object",
            "properties": {
                "score": {
                    "title": "Score",
                    "description": "complexity score of the agent",
                    "type": "integer"
                },
                "rationale": {
                    "title": "Rationale",
                    "description": "rationale for the complexity score",
                    "type": "string"
                }
            },
            "required": [
                "score",
                "rationale"
            ]
        }
    }
}
```
    """
    example_output_json = """
```json
{
    "es": [
        {"score": 1, "rationale": "rationale 1"},
        {"score": 2, "rationale": "rationale 2"},
        ...
    ]
}
```
"""
    
    template = """\
Your answer should be formatted as a JSON instance that conforms to the JSON schema.

As an example, given the JSON schema:
{example_json_schema}

Your answer in this case should be formatted as follows:
{example_output_json}

Here's the real JSON schema:
{real_json_schema}

Please provide your answer in the correct json format accordingly. Pay attention to the enum field in properties, do not generate answer that is not in the enum field if provided.
"""
    return template.format(
        example_json_schema=example_json_schema,
        example_output_json=example_output_json,
        real_json_schema=schema.schema_json()
    )
    
def inspect_input(inputs, **kwargs):
    print(inputs)
    return inputs
from langchain_core.runnables import RunnableLambda
inspect_runnable = RunnableLambda(inspect_input)

class LLMTracker(BaseCallbackHandler):
    def __init__(self, cmodule: 'LangChainLM'):
        super().__init__()
        self.cmodule = cmodule
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        meta = response.llm_output['token_usage']
        meta['model'] = response.llm_output['model_name']
        self.cmodule.llm_gen_meta.append(deepcopy(meta))

class LangChainSemantic(LMSemantic):
    """Please do not set your format instructioins in the prompt
    just specify the output format with a pydantic model (basically the dspy way)
    
    When applying optimization we need to control how to format your output schema bc 
    we might interleave the reasoning process with the output generation
    """
    def __init__(
        self,
        system_prompt: str,
        inputs: Union[str, list[str]],
        output_format: Union[Type[BaseModel], str],
        output_format_instructions: str = None,
        img_input_idices: list[int] = None,
        enable_memory: bool = False,
        input_key_in_mem: str = None,
        following_messages: list[MessageLikeRepresentation] = [],
    ):
        self.system_prompt = system_prompt
        self.img_input_idices = img_input_idices
        # NOTE: this is short-term memory, only history wihtin a workflow execution is recorded
        assert not enable_memory, "memory flag experimental, use following_messages to manage explicitly"
        self.enable_memory = enable_memory
        if enable_memory:
            assert input_key_in_mem is not None, "Must provide input_key_in_mem if enable_memory is True"
        self.input_key_in_mem = input_key_in_mem
        
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        assert len(self.inputs) > 0, "At least one input is required"
        
        self.output_format_instructions = output_format_instructions
        
        if isinstance(output_format, str):
            self.output_format = None
            self.parser = None
            self.outputs = [output_format]
        else:
            self.output_format = output_format
            self.parser = JsonOutputParser(pydantic_object=output_format)
            # NOTE: output name is inferred from the output format, use top-level fields only
            self.outputs = list(self.output_format.__fields__.keys())
            if not self.output_format_instructions:
                self.output_format_instructions = get_format_instruction(output_format)
        
        self.message_template_predefined = len(following_messages) > 0
        self.follwing_messages = following_messages
        self._chat_prompt_template: ChatPromptTemplate = None
    
    @property
    def chat_prompt_template(self):
        return self._chat_prompt_template

    @chat_prompt_template.setter
    def chat_prompt_template(self, value):
        self._chat_prompt_template = value
    
    def build_prompt_template(self):
        # setup message list
        if not self.message_template_predefined:
            user_messages = []
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
                    input_fields.append(f"- {input}:\n{{{input}}}")
            usr_prompt = "\n".join(input_fields) + "\n\n" + \
                        "Your answer:\n"
            user_messages.append(
                {
                    "type": "text",
                    "text": usr_prompt
                }
            )
            self.usr_prmopt_template = HumanMessagePromptTemplate.from_template(template=user_messages)
            self.follwing_messages = [self.usr_prmopt_template]
        # setup prompt template
        if self.enable_memory:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    MessagesPlaceholder(variable_name="compiler_chat_history"),
                ] + self.follwing_messages
            )
        else:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                ] + self.follwing_messages
            )
        if self.output_format_instructions:
            self.chat_prompt_template.append(HumanMessage(self.output_format_instructions))

    def get_agent_role(self) -> str:
        return self.system_prompt

    def get_agent_inputs(self) -> list[str]:
        return self.inputs

    def get_agent_outputs(self) -> list[str]:
        return self.outputs

    def get_output_schema(self) -> BaseModel:
        return self.output_format

    def get_high_level_info(self) -> str:
        dict = {
            "agent_prompt": self.system_prompt,
            "input_names": self.inputs,
            "output_names": self.outputs,
        }
        return json.dumps(dict, indent=4)

    def get_formatted_info(self) -> str:
        output_schemas = json.loads(self.output_format.schema_json())
        dict = {
            "agent_prompt": self.system_prompt,
            "input_varaibles": self.inputs,
            "output_json_schema": output_schemas
        }
        return json.dumps(dict, indent=4)


class LangChainLM(LLMPredictor):
    def __init__(self, name, semantic: LangChainSemantic, lm = None) -> None:
        self.llm_gen_meta = []
        self.chat_history = ChatMessageHistory()
        self.semantic: LangChainSemantic = semantic # mainly for type hint
        # default model can be overwritten by set_lm
        if lm is None:
            self.lm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.0, 
                callbacks=[LLMTracker(self)]
            )
        else:
            self.lm = lm
        super().__init__(name, semantic, self.lm)
    
    def get_invoke_routine(self):
        self.semantic.build_prompt_template() # will rebuild the prompt template for each re-compile
        inputs_str = ', '.join(self.semantic.inputs)
        invoke_arg_dict_str = '{' + ', '.join(
                [f'"{input}": {input}' for input in self.semantic.inputs] 
            ) + '}'
        #NOTE: use imperative merge at runtime bc message placeholder cannot be merged statically
        routine = self.semantic.chat_prompt_template | self.lm
        # print(self.semantic.chat_prompt_template)
        if self.semantic.enable_memory:
            routine = RunnableWithMessageHistory(
                runnable=routine,
                get_session_history=lambda: self.chat_history,
                input_messages_key=self.semantic.input_key_in_mem,
                history_messages_key="compiler_chat_history",
            )
        if self.semantic.output_format:
            result_str = '{' + ', '.join(f'"{output}": result.{output}' for output in self.semantic.outputs) + '}'
            routine = routine | self.semantic.parser
            langchain_kernel_template = f"""
def langchain_lm_kernel({inputs_str}):
    result = routine.invoke({invoke_arg_dict_str})
    result = output_format.parse_obj(result)
    # print(result)
    return {result_str}
"""
        else:
            result_str = f'{{"{self.semantic.outputs[0]}": result.content}}'
            langchain_kernel_template = f"""
def langchain_lm_kernel({inputs_str}):
    result = routine.invoke({invoke_arg_dict_str})
    return {result_str}
"""
        self.kernel_str = langchain_kernel_template
        func_obj = compile(langchain_kernel_template, '<string>', 'exec')
        local_name_space = {}
        exec(func_obj, 
             {
                'routine': routine, 
                'output_format': self.semantic.output_format,
            }, local_name_space)
        return local_name_space['langchain_lm_kernel']
    
    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        model_name: str = self.lm_config['model']
        if model_name.startswith('gpt-'):
            self.lm = ChatOpenAI(
                **self.lm_config, 
                callbacks=[LLMTracker(self)]
            )
            self.kernel = self.get_invoke_routine()
        else:
            raise ValueError(f"Model {model_name} not supported")
        return

    def get_lm_history(self):
        hist_cpy = deepcopy(self.llm_gen_meta)
        self.llm_gen_meta = []
        return hist_cpy

    def custom_reset(self):
        self.llm_gen_meta = []
        self.chat_history.clear()