SYSTEM_PROMPT = '''Finalize the code.'''
USER_HISTORY = "\nHere's what you need to know(Remember, this is just information, Try not to repeat what's inside):\nHere is the relevant history you may need:\n\nname: Alic\nrole: Code Reviewer\nspeak content: {completed_code}\nHere is the new chat history:\n\n"
USER_PROMPT = '\nAs a code master, your task is to improve the another agent\'s output code based on messages before. Your evaluation should consider syntax accuracy, logical completeness, and adherence to the initial intent of the code. If the agent\'s completion and corrections meet the required standards, output the current code as it is. If the completion or corrections do not satisfy the criteria, please provide the corrected version of the code. Please complete the code (just continue writing and put result between <result> and </result>, do not output any other content!)\n {incomplete_code} \nPlease continue the conversation on behalf of David, making your answer appear as natural and coherent as possible, and try to speak differently from what others have already said.'

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

code_finalize_semantic = LangChainSemantic(
  SYSTEM_PROMPT,
  ['completed_code', 'incomplete_code'],
  "finalized_code",
  following_messages=[("user", USER_HISTORY), ("user", USER_PROMPT)]
)

code_finalize_lm = LangChainLM('code finalize', code_finalize_semantic, opt_register=True)
code_finalize_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
  
}
code_finalize_agent = code_finalize_lm.as_runnable()
