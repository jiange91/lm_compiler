SYSTEM_PROMPT = '''
You are a Result Extractor, responsible for extracting the final result from the code.

As a code master, your task is to improve the another agent's output code based on messages before. Your evaluation should consider syntax accuracy, logical completeness, and adherence to the initial intent of the code. If the agent's completion and corrections meet the required standards, output the current code as it is. If the completion or corrections do not satisfy the criteria, please provide the corrected version of the code. Please complete the code.
'''

TRANSIT_NODE_PROMPT = '''
The nodes that can be transited to are: ['Finalize_Code', 'end_node']. Please decide the next node that should be transited to. If the current node's task of finalizing code is complete, then please consider transit to the end node. Otherwise, the task is not complete, stay at the current node.
'''

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from pydantic import BaseModel, Field

code_finalize_semantic = LangChainSemantic(
  SYSTEM_PROMPT,
  ['completed_code'],
  "finalized_code",
  output_format_instructions="Please give your completed code between <result> and </result>, do not output any other content!"
)

code_finalize_lm = LangChainLM('code finalize', code_finalize_semantic, opt_register=True)
code_finalize_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
  
}
code_finalize_agent = code_finalize_lm.as_runnable()

code_finalize_transit_semantic = LangChainSemantic(
  TRANSIT_NODE_PROMPT,
  ['completed_code'],
  "output_node",
  output_format_instructions="Give your output in format <node>node_name</node> where node_name is the name of the next node chosen from ['Finalize_Code', 'end_node']"
)

code_finalize_transit_lm = LangChainLM('finalize transit node', code_finalize_transit_semantic, opt_register=True)
code_finalize_transit_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
}
code_finalize_transit_agent = code_finalize_transit_lm.as_runnable()
