SYSTEM_PROMPT = '''
You are a Code Reviewer, responsible for reviewing the first half of the incomplete Python code to understand its structure, logic, and existing functionality. 

You are now a code master. Please complete the following additional code, ensure the function handles inputs and matches the expected outputs as described (just continue writing and put result between <result> and </result>, do not output any other content!)\n{prompt}
'''

TRANSIT_NODE_PROMPT = '''
To ensure the subtask 'Receive the first half of the incomplete Python code' is complete, define the following rules based on the output of the roles:\n\n1. **Code Reviewer Responsibilities**: The Code Reviewer must thoroughly understand the structure, logic, and existing functionality of the provided code snippet.\n\n2. **Developer Responsibilities**: The Developer must complete the second half of the code, ensuring it is functionally correct and integrates seamlessly with the first half.\n\nIf all the rules above are met, the task is complete.
The nodes that can be transited to are: ['Receive_Incomplete_Code', 'Finalize_Code']. Please decide the next node that should be transited to, and finally, output <node>node_name</node> where node_name is the name of the next node. Let's think step by step.
'''

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from pydantic import BaseModel, Field

code_completion_semantic = LangChainSemantic(
  SYSTEM_PROMPT,
  ['prompt'],
  "completed_code",
)

code_completion_lm = LangChainLM('code completion', code_completion_semantic, opt_register=True)
code_completion_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
}
code_completion_agent = code_completion_lm.as_runnable()

code_completion_transit_semantic = LangChainSemantic(
  TRANSIT_NODE_PROMPT,
  ['code'],
  "output_node",
)

code_completion_transit_lm = LangChainLM('transit node', code_completion_transit_semantic, opt_register=True)
code_completion_transit_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
}
code_completion_transit_agent = code_completion_transit_lm.as_runnable()

