from compiler.optimizer.params import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor 


class LMReasoning(ParamBase):
    level = ParamLevel.NODE
    
from langchain_core.messages import HumanMessage

class ZeroShotCoT(OptionBase):
    def apply(self, lm_module: LLMPredictor):
        suffix = HumanMessage("Reasoning: Let's think step by step: \n")
        lm_module.semantic.append_following_messages([suffix])
        # new prompt template will be reset at outside
        return lm_module

class PlanBefore(OptionBase):
    def apply(self, lm_module: LLMPredictor):
        suffix = HumanMessage("Reasoning: Let's first plan necessary steps to approach this problem then give the answer: \n")
        lm_module.semantic.append_following_messages([suffix])
        # new prompt template will be reset at outside
        return lm_module

