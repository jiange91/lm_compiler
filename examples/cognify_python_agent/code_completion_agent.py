CORRECT_SYSTEM_PROMPT = """
Your task is to read incomplete Python functions and complete them based on the provided docstring.
"""

from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM, LMConfig
from pydantic import BaseModel, Field
from compiler.optimizer.params.reasoning import ZeroShotCoT


code_completion_semantic = LangChainSemantic(
  CORRECT_SYSTEM_PROMPT,
  ['incomplete_function'],
  "completed_code",
  output_format_instructions="Please response with the function body only. Place the code within <result> and </result> tags. Do not include any additional text, explanations, or comments outside these tags",
)

lm_config = LMConfig(
  provider="openai",
  model="gpt-4o-mini",
  kwargs={"temperature": 0.0},
)
code_completion_lm = LangChainLM(
  'code completion', 
  code_completion_semantic, 
  opt_register=True, 
  lm_config=lm_config,
)
# ZeroShotCoT.direct_apply(code_completion_lm)
code_completion_agent = code_completion_lm.as_runnable()
