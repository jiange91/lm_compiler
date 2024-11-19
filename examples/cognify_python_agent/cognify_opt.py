from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.scaffolding import LMScaffolding
from cognify.hub.cogs import reasoning, model_selection, common
from cognify.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from cognify.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from cognify.langchain_bridge.interface import LangChainLM
from cognify.hub.cogs import ensemble
import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna

from cognify.optimizer.registry import register_data_loader

def raw_test(data):
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=data,
        n_parallel=50,
    )
    eval_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/cognify_python_agent/workflow.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    print(evaluator.get_score('test', eval_task, show_process=True))

from data_loader import load_data   

if __name__ == '__main__':
    train, val, dev = load_data()
    raw_test(dev)