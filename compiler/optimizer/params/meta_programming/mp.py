
question_suffix_or_path: str = "\n\nLet's first come up with a list of experts you may want to consult for this problem and then immediately start solving it."  # "" # "\n\nLet's think step by step."
intermediate_feedback = "Based on the information given, what are the most logical next steps or conclusions? Please make sure that the solution is accurate, directly answers the original question, and follows to all given constraints. Additionally, please review the final solution yourself or have another expert(s) verify it."

expert_python_message: str = 'You are an expert in Python and can generate Python code. To execute the code and display its output in the terminal using print statements, please make sure to include "Please run this code!" after the code block (i.e., after the closing code blocks)'

import os
import json

script_dir = os.path.dirname(__file__)

meta_config_path = os.path.join(script_dir, "prompts", "meta-v0-2023-08-14-baseline.json")
with open(meta_config_path, "r") as f:
    meta_prompt_config_dict = json.load(f)
meta_model_message_list = meta_prompt_config_dict["meta-model"]["message-list"]

question_prefix_path = os.path.join(script_dir, "prompts", "meta-prompting-with-no-python-expert-instruction.txt")
with open(question_prefix_path, "r") as f:
    question_prefix = f.read()
question_prefix_or_path = question_prefix

meta_model_settings = meta_prompt_config_dict["meta-model"]
generator_settings = meta_prompt_config_dict["generator"]
verifier_settings = meta_prompt_config_dict["verifier"]
summarizer_settings = meta_prompt_config_dict["summarizer"]

# Get the error message and final answer indicator
error_message = meta_prompt_config_dict["meta-model"]["error-message"]
final_answer_indicator = meta_prompt_config_dict["meta-model"][
    "final-answer-indicator"
]


from compiler.optimizer.params import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor 
from compiler.langchain_bridge.interface import LLMTracker, LangChainLM
from langchain_openai import ChatOpenAI


class MetaPrompting(OptionBase):
    def apply(self, lm_module: LangChainLM):
        task_description = lm_module.semantic.get_agent_role()
        lm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.0, 
            callbacks=[LLMTracker(lm_module)]
        ) 
        return lm_module