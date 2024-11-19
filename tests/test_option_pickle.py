import sys
import os
import json
import random
import uuid
import pickle

import multiprocessing as mp
from cognify.graph.program import StatePool, Module
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.optimizer.evaluation.evaluator import Evaluator
from cognify.optimizer.evaluation.metric import MetricBase, MInput
from cognify.llm import CogLM, InputVar, OutputLabel
from cognify.llm.model import LMConfig

from cognify.hub.cogs import reasoning, model_selection, common, ensemble
from cognify.hub.cogs.utils import load_params
from cognify.hub.cogs.reasoning import ZeroShotCoT, PlanBefore

from cognify.utils import load_api_key
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

lm = CogLM('qa_agent', system_prompt="Repeat the input", input_variables=[InputVar(name='input')], 
           output=OutputLabel(name='answer'), 
           lm_config=LMConfig(model="gpt-4o-mini", kwargs={'temperature': 0.0}))
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