from langchain_openai import ChatOpenAI
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
from typing import Union, Callable
import types
from copy import deepcopy
import json

from compiler.IR.llm import LLMPredictor, LMConfig, LMSemantic

logger = logging.getLogger(__name__)

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
    ):
        self.system_prompt = system_prompt
        self.inputs = inputs if isinstance(inputs, list) else [inputs]
        self.output_format = output_format
        # NOTE: output name is inferred from the output format, use top-level fields only
        self.outputs = list(self.output_format.__fields__.keys())
        
        input_fields = []
        for input in self.inputs:
            input_fields.append(f"- {input}:\n{{{input}}}")
        usr_prompt = "\n".join(input_fields) + "\n\nYour answer:\n"
        
        self.chat_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", usr_prompt),
            ]
        )
        
        self.langchain_lm_kernel = self.create_kernel_func()
    
    def create_kernel_func(self):
        inputs_str = ', '.join(self.inputs)
        invoke_arg_dict_str = '{' + ', '.join([f'"{input}": {input}' for input in self.inputs]) + '}'
        result_str = '{' + ', '.join(f'"{output}": result.{output}' for output in self.outputs) + '}'
        
        langchain_lm_template = f"""
def langchain_lm_kernel(llm, {inputs_str}):
    sllm = llm.with_structured_output(self.output_format)
    routine = self.chat_prompt_template | sllm
    result = routine.invoke({invoke_arg_dict_str})
    return {result_str}
            """
        self.kernel_str = langchain_lm_template
        func_obj = compile(langchain_lm_template, '<string>', 'exec')
        local_name_space = {}
        exec(func_obj, {'self': self}, local_name_space)
        return local_name_space['langchain_lm_kernel']

    def get_agent_role(self) -> str:
        return self.system_prompt

    def get_agent_inputs(self) -> list[str]:
        return self.inputs

    def get_agent_outputs(self) -> list[str]:
        return self.outputs

    def get_invoke_routine(self):
        return self.langchain_lm_kernel

    def get_output_schema(self) -> BaseModel:
        return self.output_format

    def get_formatted_info(self) -> str:
        output_schemas = json.loads(self.output_format.schema_json())
        dict = {
            "agent_prompt": self.system_prompt,
            "input_varaibles": self.inputs,
            "output_json_schema": output_schemas
        }
        return json.dumps(dict, indent=4)

class LangChainLM(LLMPredictor):
    def __init__(self, name, semantic: LangChainSemantic) -> None:
        super().__init__(name, semantic)
        self.llm_gen_meta = []
    
    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        model_name: str = self.lm_config['model']
        if model_name.startswith('gpt-'):
            self.lm = ChatOpenAI(**self.lm_config, callbacks=[LLMTracker(self)])
        else:
            raise ValueError(f"Model {model_name} not supported")
        return 

    def get_lm_history(self):
        hist_cpy = deepcopy(self.llm_gen_meta)
        self.llm_gen_meta = []
        return hist_cpy

if __name__ == "__main__":
    class ComplexityEstimation(BaseModel):
        """complexity of each agent"""
        score: int = Field(
            description="complexity score of the agent"
        )
        rationale: str = Field(
            description="rationale for the complexity score"
        )

    class ComplexityList(BaseModel):
        """complexity of all agents"""
        es: list[ComplexityEstimation] = Field(
            description="complexity of all agents"
        )
        
    complexity_system = """
    You are an expert at designing LLM-based agent workflow. Your task is to evaluate the complexity of the responsibility of each agent. 

    You will be provided with their system prompts for reference. Please assign each agent a numerical rating on the following scale to indicate the complexity. You should consider:
    Does the task require multi-step reasoning, planning, decision-making? 
    Does the task encompass a wide range of responsibilities?
    For each agent, please give your rate from 1 to 5 and provide your rationale for the rating.

    Rating criteria: 1: straightforward, 5: very complex. 

    Please follow the order of the given agents.
    """

    semantic = LangChainSemantic(
        system_prompt=complexity_system,
        inputs=['agent_prompts'],
        output_format=ComplexityList
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.75)
    print(semantic.get_invoke_routine()(llm, agent_prompts = ['You are an expert at writing a list of short passages given a user query. You should give sub-queries that cover comprehensive aspects of the original query and should not have too many overlaps. Each sub-query should be followed by a short passage that answers that topic.']))