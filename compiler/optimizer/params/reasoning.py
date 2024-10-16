from abc import ABC, ABCMeta

from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase, IdentityOption
from compiler.IR.llm import LLMPredictor
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, get_inspect_runnable 
from compiler.IR.program import Workflow
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
import copy
import types
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables import chain

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

def format_rationale(rationale: list[BaseMessage]) -> list[BaseMessage]:
    base = HumanMessage("Here is the reasoning steps:\n")
    return [base] + rationale

@chain
def pass_msgs(msgs: list[BaseMessage]):
    return msgs

class ReasonThenFormat(OptionBase, metaclass=ReasoningOptionMeta):
    
    @classmethod
    def direct_apply(cls, lm_module: LangChainLM):
        reasoning = cls()
        reasoning.apply(lm_module)
        return reasoning

    def reasoning_step(
        self, 
        chat_messages: list[BaseMessage], 
        lm: ChatOpenAI, 
    ) -> list[BaseMessage]:
        """Produce reasoning steps for the given chat prompt messages
        """
        raise NotImplementedError

    def get_invoke_routine(self, lm_module: LangChainLM):
        """
        If the orignal output has certain format, applying additional reasoning steps will break down
        it into two phases, first one allows free generation along with reasoning steps, and the second
        one will the formatting step
        
        this is the helper functor to facilitate this process
        """
        old_semantic = lm_module.semantic
        # remove format instruction
        following_messages = old_semantic.following_messages.copy() if old_semantic.message_template_predefined else []
        new_semantic = LangChainSemantic(
            system_prompt=old_semantic.system_prompt,
            inputs=old_semantic.inputs,
            output_format=old_semantic.outputs[0],
            need_output_type_hint=False,
            output_format_instructions=None,
            img_input_idices=old_semantic.img_input_idices,
            enable_memory=old_semantic.enable_memory,
            input_key_in_mem=old_semantic.input_key_in_mem,
            following_messages=following_messages,
            demos=old_semantic.demos,
        )
        new_semantic.build_prompt_template()
        
        def new_invocation_routine(lm_module: LangChainLM):
            def get_answer(inputs: dict):
                chat_messages = new_semantic.chat_prompt_template.format_messages(**inputs)
                chat_messages += [
                    HumanMessage("Don't give your final response to the instruction directly. We can start with some reasoning first.\n")
                ]
                
                # get reasoning steps, avoid directly modifying the original chat_messages
                rationale = self.reasoning_step(chat_messages.copy(), lm_module.lm)
                
                # NOTE: this will be cleared after current invocation in compiler/IR/llm.py
                #       incase a module is reused
                rationale_str = get_buffer_string(rationale)
                lm_module.rationale = rationale_str
                
                # prepare output propmt
                chat_messages.extend(rationale)
                post_reasoning_routine = pass_msgs | get_inspect_runnable(f'-- {lm_module.name} reasoning steps --') | lm_module.lm | get_inspect_runnable(f'-- {lm_module.name} organize output --')
                
                output_vars = ', '.join([f'{{{ovar}}}' for ovar in old_semantic.get_agent_outputs()])
                if old_semantic.output_type_hint or old_semantic.output_format_instructions:
                    chat_messages.extend([
                        HumanMessage(f"Now please `{old_semantic.outputs[0]}` according to the following instructions:\n"),
                        old_semantic.get_output_format_spec(),
                    ])
                    if old_semantic.output_format:
                        post_reasoning_routine = post_reasoning_routine | old_semantic.parser
                        try:
                            # print("if old_semantic.output_format")
                            result = post_reasoning_routine.invoke(chat_messages)
                        except Exception as e:
                            print(e)
                            print("ERR IN new_invocation_routine if old_semantic.output_format")
                            print(chat_messages)
                            print(post_reasoning_routine)
                        result = old_semantic.output_format.model_validate(result) 
                        return {key: getattr(result, key) for key in old_semantic.get_agent_outputs()}
                    else:
                        try:
                            # print("else old_semantic.output_format")
                            result = post_reasoning_routine.invoke(chat_messages).content
                        except Exception as e:
                            print(e)
                            print("ERR IN new_invocation_routine else not old_semantic.output_format")
                            print(chat_messages)
                            print(post_reasoning_routine)
                        return {old_semantic.get_agent_outputs()[0]: result}
                else:
                    assert len(old_semantic.get_agent_outputs()) == 1, "No formatted agent only has one output"
                    chat_messages.append(
                        HumanMessage(f"Based on these informations, please provide`{old_semantic.outputs[0]}` directly.")
                    )
                    try:
                        # print("not old_semantic.output_type_hint")
                        result = post_reasoning_routine.invoke(chat_messages).content
                    except Exception as e:
                        print(e)
                        print("ERR IN new_invocation_routine else not old_semantic.output_type_hint")
                        print(chat_messages)
                        print(post_reasoning_routine)
                    return {old_semantic.get_agent_outputs()[0]: result}
            
            inputs_str = ', '.join(new_semantic.inputs)
            invoke_arg_dict_str = '{' + ', '.join(
                    [f'"{input}": {input}' for input in new_semantic.inputs] 
                ) + '}'
            new_kernel_str = f"""
def langchain_lm_kernel({inputs_str}):
    answer = get_answer({invoke_arg_dict_str})
    return answer
            """
            lm_module.kernel_str = new_kernel_str
            func_obj = compile(new_kernel_str, '<string>', 'exec')
            local_name_space = {}
            exec(func_obj, 
                {
                    'get_answer': get_answer, 
                }, local_name_space)
            return local_name_space['langchain_lm_kernel']
        return new_invocation_routine(lm_module)
    
    def apply(self, lm_module: LangChainLM):
        # lm_module.get_invoke_routine = types.MethodType(new_invocation_routine, lm_module)
        lm_module.reasoning = self
        lm_module.lm = None # to trigger reset() incase you forget
        return lm_module
        

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

class ZeroShotCoT(ReasonThenFormat):
    def __init__(self):
        super().__init__("ZeroShotCoT")
        self.cost_indicator = 4.0
        
    def reasoning_step(
        self, 
        chat_messages: list[BaseMessage], 
        lm: ChatOpenAI | ChatTogether, 
    ) -> list[BaseMessage]:
        h = HumanMessage("Let's solve this problem step by step before giving the final response\n")
        chat_messages.append(h)
        try:
            # print("zero shot reasoning step")
            result = lm.invoke(chat_messages)
        except Exception as e:
            print(e)
            print("ERR IN zero-shot reasoning step")
            print(chat_messages)
            print(lm)
        logger.debug(f"Zero-shot CoT in module {lm.name}, reasoning: {result.content}")
        return [h, result]
        

class PlanBefore(ReasonThenFormat):
    def __init__(self):
        super().__init__("PlanBefore")
        self.cost_indicator = 2.5
    
    def reasoning_step(
        self, 
        chat_messages: list[BaseMessage], 
        lm: ChatOpenAI, 
    ) -> list[BaseMessage]:
        h = HumanMessage("Let's first plan necessary steps to approach this problem before giving the final response\n")
        chat_messages.append(h)
        result = lm.invoke(chat_messages)
        logger.debug(f"PlanBefore in module {lm.name}, reasoning: {result.content}")
        return [h, result]
        