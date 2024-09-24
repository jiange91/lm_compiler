from .prompt import SYSTEM_PROMPT, EXPERT_USER_PROMPT
from agents.openai_chatComplete import completion_with_backoff, completion_with_log
from agents.utils import fill_in_placeholders, get_error_message, is_run_code_success, print_chat_message


class QueryExpansionAgent():
    def __init__(self, expert_ins, simple_ins,model_type='gpt-4'):
        self.chat_history = []
        self.expert_ins = expert_ins
        self.simple_ins = simple_ins
        self.model_type = model_type

    def run(self, query_type):
        if query_type == 'expert':
            information = {
                'query': self.expert_ins,
            }
        else:
            information = {
                'query': self.simple_ins,
            }

        messages = []
        messages.append({"role": "system", "content": fill_in_placeholders(SYSTEM_PROMPT, information)})
        messages.append({"role": "user", "content": fill_in_placeholders(EXPERT_USER_PROMPT, information)})
        expanded_query_instruction = completion_with_log(messages, self.model_type)

        return expanded_query_instruction

import dspy
from agents.dspy_common import OpenAIModel
from agents.config.openai import openai_kwargs

class QueryExpansion(dspy.Signature):
    """According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. List all the appropriate libraries and pinpoint the correct library functions to call and set each parameter in every function call accordingly.

    You should understand what the query's requirements are, and output step by step, detailed instructions on how to use python code to fulfill these requirements. Include what libraries to import, what library functions to call, how to set the parameters in each function correctly, how to prepare the data, how to manipulate the data so that it becomes appropriate for later functions to call etc,. Make sure the code to be executable and correctly generate the desired output in the user query. 
    """
    
    query = dspy.InputField(format=str)
    expanded_query = dspy.OutputField(format=str)

class QueryExpansionModule(dspy.Module):
    def __init__(self, model_type='gpt-4o-mini'):
        self.model_type = model_type
        self.chat_history = []
        self.engine = OpenAIModel(
            model=model_type, **openai_kwargs
        )

        self.expand_query = dspy.Predict(QueryExpansion)
        
    def forward(
        self,
        query,
    ):
        with dspy.settings.context(lm=self.engine):
            expanded_query_instruction = self.expand_query(query=query)
            return expanded_query_instruction
    
    def run(
        self,
        query,
    ):
        expanded_query_instruction = self.forward(query)
        return expanded_query_instruction.expanded_query