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

SYSTEM_PROMPT = '''
According to the user query, expand and solidify the query into a step by step detailed instruction (or comment) on how to write python code to fulfill the user query's requirements. List all the appropriate libraries and pinpoint the correct library functions to call and set each parameter in every function call accordingly.

You should understand what the query's requirements are, and output step by step, detailed instructions on how to use python code to fulfill these requirements. Include what libraries to import, what library functions to call, how to set the parameters in each function correctly, how to prepare the data, how to manipulate the data so that it becomes appropriate for later functions to call etc,. Make sure the code to be executable and correctly generate the desired output in the user query. 
'''

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from pydantic import BaseModel, Field

query_expansion_semantic = LangChainSemantic(
    SYSTEM_PROMPT,
    ['query'],
    "expanded_query",
)

query_expansion_lm = LangChainLM('query expansion', query_expansion_semantic, opt_register=True)
query_expansion_lm.lm_config = {
    'model': 'gpt-4o-mini',
    'temperature': 0.0,
}
query_expansion_agent = query_expansion_lm.as_runnable()