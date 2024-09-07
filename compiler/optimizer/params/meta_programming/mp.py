
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

question_prefix_path = os.path.join(script_dir, "prompts", "meta-prompting-instruction.txt")
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


from compiler.optimizer.params.reasoning import ReasonThenFormat
from compiler.IR.llm import LLMPredictor 
from compiler.langchain_bridge.interface import LLMTracker, LangChainLM, LangChainSemantic
from .helper import MetaPromptingScaffolding
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate

meta_model = MetaPromptingScaffolding(
    generator_settings=generator_settings,
    verifier_settings=verifier_settings,
    summarizer_settings=summarizer_settings,
    error_message=error_message,
    final_answer_indicator=final_answer_indicator,
    expert_python_message=expert_python_message,
    intermediate_feedback=intermediate_feedback,
    fresh_eyes=True,
    include_expert_name_in_instruction=True,
    extract_output=False,
    use_zero_shot_cot_in_expert_messages=False,
)

class MetaPrompting(ReasonThenFormat):
    def reasoning_step(self, new_semantic: LangChainSemantic, lm: ChatOpenAI, inputs: dict):
        chat_prompt: list[BaseMessage] = new_semantic.chat_prompt_template.format_messages(**inputs)
        chat_prompt = merge_message_runs(chat_prompt)
        
        chat_prompt.insert(max(len(chat_prompt)-1, 0), HumanMessage(question_prefix_or_path))
        chat_prompt.append(HumanMessage(question_suffix_or_path))
        
        reasoning_history = meta_model.meta_model_generate(lm, chat_prompt)
        new_semantic.chat_prompt_template = ChatPromptTemplate.from_messages(reasoning_history)