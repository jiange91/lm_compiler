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

import logging
from typing import Union, Callable
import types
from copy import deepcopy
import json

from compiler.IR.llm import LLMPredictor, LMConfig, LMSemantic

logger = logging.getLogger(__name__)

def get_format_instruction(schema: BaseModel):
    example_json_schema = """
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
    """
    example_output_json = """
{
    "es": [
        {"score": 1, "rationale": "rationale 1"},
        {"score": 2, "rationale": "rationale 2"},
        ...
    ]
}
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
    

class LLMTracker(BaseCallbackHandler):
    def __init__(self, cmodule: 'LangChainLM'):
        super().__init__()
        self.cmodule = cmodule
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        meta = response.llm_output['token_usage']
        meta['model'] = response.llm_output['model_name']
        self.cmodule.llm_gen_meta.append(deepcopy(meta))

class LangChainSemantic(LMSemantic):
    def __init__(
        self,
        system_prompt: str,
        inputs: Union[str, list[str]],
        output_format: BaseModel,
        img_input_idices: list[int] = None,
        enable_memory: bool = False,
        input_key_in_mem: str = None,
    ):
        self.system_prompt = system_prompt
        self.img_input_idices = img_input_idices
        # NOTE: this is short-term memory, only history wihtin a workflow execution is recorded
        self.enable_memory = enable_memory
        if enable_memory:
            assert input_key_in_mem is not None, "Must provide input_key_in_mem if enable_memory is True"
        self.input_key_in_mem = input_key_in_mem
        
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        assert len(self.inputs) > 0, "At least one input is required"
        self.output_format = output_format
        self.parser = JsonOutputParser(pydantic_object=output_format)
        # NOTE: output name is inferred from the output format, use top-level fields only
        self.outputs = list(self.output_format.__fields__.keys())
        
        # Set prompt template
        self.system_prompt_template = self.system_prompt + "\n\n" + "{there_is_no_way_overlap_output_format}"
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
        
        # default model can be overwritten by set_lm
        if lm is None:
            self.lm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.0, 
                callbacks=[LLMTracker(self)]
            )
        else:
            self.lm = lm
        
        if semantic.enable_memory:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", semantic.system_prompt_template),
                    MessagesPlaceholder(variable_name="chat_history"),
                    semantic.usr_prmopt_template,
                ]
            ).partial(there_is_no_way_overlap_output_format=get_format_instruction(semantic.output_format))
        else:
            self.chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", semantic.system_prompt_template),
                    semantic.usr_prmopt_template,
                ]
            ).partial(there_is_no_way_overlap_output_format=get_format_instruction(semantic.output_format))
        
        super().__init__(name, semantic, self.lm)
    
    def get_invoke_routine(self, semantic: LangChainSemantic):
        inputs_str = ', '.join(semantic.inputs)
        invoke_arg_dict_str = '{' + ', '.join(
                [f'"{input}": {input}' for input in semantic.inputs] 
            ) + '}'
        result_str = '{' + ', '.join(f'"{output}": result.{output}' for output in semantic.outputs) + '}'
        
        routine = self.chat_prompt_template | self.lm
        if semantic.enable_memory:
            routine = RunnableWithMessageHistory(
                runnable=routine,
                get_session_history=lambda: self.chat_history,
                input_messages_key=semantic.input_key_in_mem,
                history_messages_key="chat_history",
            )
        routine_structured_output = routine | semantic.parser
        langchain_kernel_template = f"""
def langchain_lm_kernel({inputs_str}):
    result = routine_structured_output.invoke({invoke_arg_dict_str})
    result = output_format.parse_obj(result)
    return {result_str}
"""
        self.kernel_str = langchain_kernel_template
        func_obj = compile(langchain_kernel_template, '<string>', 'exec')
        local_name_space = {}
        exec(func_obj, 
             {
                'routine_structured_output': routine_structured_output, 
                'output_format': semantic.output_format
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
            self.kernel = self.get_invoke_routine(self.semantic)
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