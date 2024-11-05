import sys
import os
import json
import random
import uuid
import pickle

import multiprocessing as mp
from compiler.IR.program import StatePool, Module
from compiler.cog_hub.fewshot import LMFewShot
from compiler.optimizer.evaluation.evaluator import Evaluator
from compiler.optimizer.evaluation.metric import MetricBase, MInput
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic

from compiler.cog_hub import reasoning, model_selection, common, ensemble
from compiler.cog_hub.utils import load_params
from compiler.cog_hub.reasoning import ZeroShotCoT, PlanBefore
from compiler.IR.schema_parser import json_schema_to_pydantic_model, get_pydantic_format_instruction

from compiler.utils import load_api_key, get_bill
load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

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
    system_prompt="Repeat the input",
    inputs=["input"],
    output_format="answer",
)

lm = LangChainLM('qa_agent', semantic)
lm.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}

is_picklable(lm)

# reasoning_param = reasoning.LMReasoning(
#     "reasoning", [common.NoChange(), ZeroShotCoT(), PlanBefore()]
# )

# reasoning_param.apply_option('ZeroShotCoT', lm)

usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
ensemble_lm = usc_ensemble.apply(lm)
is_picklable(ensemble_lm)

def run(input, module: Module):
    print("running...")
    module.invoke(input)
    return "finished"

if __name__ == "__main__":
    mp.set_start_method('spawn')
    input = StatePool()
    input.init({"input": "Hello, world!"})
    tasks = []
    with mp.Pool(processes=2) as pool:
        tasks.append(pool.apply_async(run, args=(input, ensemble_lm)))
        results = [task.get() for task in tasks]
    print(results)