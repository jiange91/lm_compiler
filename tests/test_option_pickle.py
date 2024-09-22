import sys
import os
import json
import random
import uuid
import pickle


from compiler.IR.program import StatePool
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic

from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.params.utils import load_params
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.IR.schema_parser import json_schema_to_pydantic_model, get_pydantic_format_instruction

def is_picklable(obj):
    """Check if an object (e.g., a function or method) can be pickled."""
    try:
        sobj = pickle.dumps(obj)
        dsobj = pickle.loads(sobj)
        return dsobj
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        print(f"Cannot pickle {obj}: {e}")
        return None

semantic = LangChainSemantic(
    system_prompt="",
    inputs=["input"],
    output_format="answer",
)

lm = LangChainLM('qa_agent', semantic)

is_picklable(lm)

# reasoning_param = reasoning.LMReasoning(
#     "reasoning", [common.IdentityOption(), ZeroShotCoT(), PlanBefore()]
# )

# reasoning_param.apply_option('ZeroShotCoT', lm)

from typing import List

from pydantic import BaseModel, Field


with open('plan.json', 'r') as f:
    schema = json.load(f)

output_model = json_schema_to_pydantic_model(schema, '/mnt/ssd4/lm_compiler/trace_log/plan.py')

lm.semantic.output_format = output_model

new_lm: LangChainLM = is_picklable(lm)

pass