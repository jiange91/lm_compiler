system_prompt = """
You are a code expert. Given an incomplete function and the function body generated by another agent, your task is to evaluate and improve the function body, ensuring it meets the criteria.
"""

from cognify.llm import CogLM, InputVar, OutputLabel, LMConfig
from cognify.cog_hub.reasoning import ZeroShotCoT

lm_config = LMConfig(
  custom_llm_provider="openai",
  model="gpt-4o-mini",
  kwargs={"temperature": 0.0},
)

code_finalize_agent = CogLM(agent_name='code finalize', system_prompt=system_prompt,
                            input_variables=[InputVar(name='incomplete_function'), InputVar(name='completed_code')],
                            output=OutputLabel(name='finalized_code', custom_output_format_instructions='Please response with the function body only. Place the code within <result> and </result> tags. Do not include any additional text, explanations, or comments outside these tags.'),
                            lm_config=lm_config)