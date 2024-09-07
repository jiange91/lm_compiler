from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, inspect_runnable
from compiler.IR.program import Workflow
from langchain_core.messages import merge_message_runs, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
import copy
import types
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class LMReasoning(ParamBase):
    level = ParamLevel.NODE
    

class ReasonThenFormat(OptionBase):
    """
    If the orignal output has certain format, applying additional reasoning steps will break down
    it into two phases, first one allows free generation along with reasoning steps, and the second
    one will the formatting step
    
    this is the helper functor to facilitate this process
    """
    def apply(self, lm_module: LangChainLM):
        old_semantic = lm_module.semantic
        # remove format instruction
        new_semantic = LangChainSemantic(
            system_prompt=old_semantic.system_prompt,
            inputs=old_semantic.inputs,
            output_format=f"reasoning_answer_{lm_module.name}",
            output_format_instructions=None,
            img_input_idices=old_semantic.img_input_idices,
            enable_memory=old_semantic.enable_memory,
            input_key_in_mem=old_semantic.input_key_in_mem,
            following_messages=old_semantic.follwing_messages.copy(),
        )
        new_semantic.build_prompt_template()
        
        def new_invocation_routine(lm_module: LLMPredictor):
            def get_answer(inputs: dict):
                self.reasoning_step(new_semantic, lm_module.lm, inputs)
                if old_semantic.output_format_instructions:
                    new_semantic.chat_prompt_template.extend([
                        HumanMessage("Now please format your final answer according to the follwoing instructions:\n"),
                        HumanMessage(old_semantic.output_format_instructions)
                    ])
                post_reasoning_routine = new_semantic.chat_prompt_template | merge_message_runs() | lm_module.lm
                if old_semantic.output_format_instructions:
                    new_semantic.chat_prompt_template.extend([
                        HumanMessage("Now please format your final answer according to the follwoing instructions:\n"),
                        HumanMessage(old_semantic.output_format_instructions)
                    ])
                    if old_semantic.output_format:
                        post_reasoning_routine = post_reasoning_routine | old_semantic.parser
                        result = post_reasoning_routine.invoke(inputs)
                        result = old_semantic.output_format.parse_obj(result) 
                        return {key: getattr(result, key) for key in old_semantic.get_agent_outputs()}
                    else:
                        result = post_reasoning_routine.invoke(inputs).content
                        return {old_semantic.get_agent_outputs()[0]: result}
            
            inputs_str = ', '.join(new_semantic.inputs)
            invoke_arg_dict_str = '{' + ', '.join(
                    [f'"{input}": {input}' for input in new_semantic.inputs] 
                ) + '}'
            new_kernel_str = f"""
def langchain_lm_kernel({inputs_str}):
    return get_answer({invoke_arg_dict_str})
            """
            lm_module.kernel_str = new_kernel_str
            func_obj = compile(new_kernel_str, '<string>', 'exec')
            local_name_space = {}
            exec(func_obj, 
                {
                    'get_answer': get_answer, 
                }, local_name_space)
            return local_name_space['langchain_lm_kernel']

        lm_module.get_invoke_routine = types.MethodType(new_invocation_routine, lm_module)
        return lm_module
        
    def reasoning_step(self, new_semantic: LangChainSemantic, lm: ChatOpenAI, inputs: dict):
        raise NotImplementedError


class ZeroShotCoT(ReasonThenFormat):
    def reasoning_step(self, new_semantic: LangChainSemantic, lm: ChatOpenAI, inputs: dict):
        new_semantic.chat_prompt_template.append(
            HumanMessage("Reasoning: Let's solve this problem step by step: \n")
        )
        routine = new_semantic.chat_prompt_template | merge_message_runs() | lm
        result = routine.invoke(inputs).content
        new_semantic.chat_prompt_template.append(
            AIMessage(result)
        )
        

class PlanBefore(ReasonThenFormat):
    def reasoning_step(self, new_semantic: LangChainSemantic, lm: ChatOpenAI, inputs: dict):
        new_semantic.chat_prompt_template.append(
            HumanMessage("Reasoning: Let's first plan necessary steps to approach this problem then give the answer: \n")
        )
        routine = new_semantic.chat_prompt_template | merge_message_runs() | lm
        result = routine.invoke(inputs).content
        new_semantic.chat_prompt_template.append(
            AIMessage(result)
        )

