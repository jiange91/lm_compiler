import os
import sys
import json

import dspy

openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'api_provider': os.getenv('OPENAI_API_TYPE'),
    'temperature': 1.0,
    'top_p': 0.9,
}

with open('/Users/jiiiiin/wuklab/llm-compiler/compillm/storm_src/storm_lm_config.json') as f:
    lm_configs = json.load(f)

find_related_topic_lm = dspy.OpenAI(model=lm_configs['find_related_topic_lm'], max_tokens=500, **openai_kwargs)
gen_persona_lm = dspy.OpenAI(model=lm_configs['gen_persona_lm'], max_tokens=500, **openai_kwargs)
query_designer_lm = dspy.OpenAI(model=lm_configs['query_designer_lm'], max_tokens=300, **openai_kwargs)
conv_simulator_lm = dspy.OpenAI(model=lm_configs['conv_simulator_lm'], max_tokens=500, **openai_kwargs)
question_asker_lm = dspy.OpenAI(model=lm_configs['question_asker_lm'], max_tokens=500, **openai_kwargs)
outline_gen_lm = dspy.OpenAI(model=lm_configs['outline_gen_lm'], max_tokens=400, **openai_kwargs)
direct_outline_lm = dspy.OpenAI(model=lm_configs['direct_outline_lm'], max_tokens=400, **openai_kwargs)
article_gen_lm = dspy.OpenAI(model=lm_configs['article_gen_lm'], max_tokens=700, **openai_kwargs)
write_lead_lm = dspy.OpenAI(model=lm_configs['write_lead_lm'], max_tokens=4000, **openai_kwargs)
article_polish_lm = dspy.OpenAI(model=lm_configs['article_polish_lm'], max_tokens=4000, **openai_kwargs)
