completion_system_prompt = """
Your task is to read incomplete Python functions and complete them based on the provided docstring.
"""

from cognify.llm import CogLM, InputVar, OutputLabel, LMConfig
from pydantic import BaseModel, Field
from cognify.cog_hub.reasoning import ZeroShotCoT


lm_config = LMConfig(
  custom_llm_provider="openai",
  model="gpt-4o-mini",
  kwargs={"temperature": 0.0},
)
code_completion_agent = CogLM(agent_name='code completion', system_prompt=completion_system_prompt,
                              input_variables=[InputVar(name='incomplete_function')],
                              output=OutputLabel(name='completed_code', custom_output_format_instructions='Please response with the function body only. Place the code within <result> and </result> tags. Do not include any additional text, explanations, or comments outside these tags.'),
                              lm_config=lm_config)