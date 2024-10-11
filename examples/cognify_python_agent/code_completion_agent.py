CORRECT_SYSTEM_PROMPT = "Receive the first half of the incomplete Python code."
USER_PROMPT = "\nYou are now a code master. Please complete the following additional code, ensure the function handles inputs and matches the expected outputs as described (just continue writing and put result between <result> and </result>, do not output any other content!){incomplete_code}"


from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from pydantic import BaseModel, Field

code_completion_semantic = LangChainSemantic(
  CORRECT_SYSTEM_PROMPT,
  ['incomplete_code'],
  "completed_code",
)

code_completion_lm = LangChainLM('code completion', code_completion_semantic, opt_register=True)
code_completion_lm.lm_config = {
  'model': 'gpt-4o-mini',
  'temperature': 0.0,
}
code_completion_agent = code_completion_lm.as_runnable()
