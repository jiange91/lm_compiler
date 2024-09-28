from compiler.optimizer.layered_optimizer_pluggable import InnerLoopBayesianOptimization, OuterLoopOptimization
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.importance_eval_new import LMImportanceEvaluator
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask
import runpy
import uuid
import multiprocessing as mp
import json
import os
import random
import optuna

from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema
from humaneval.humaneval import HumanEvalDataset

def opt(train):
  lm_options = [
    'gpt-4o-mini',
  ]

  model_param = model_selection.LMSelection('lm_model', model_selection.model_option_factory(lm_options))
  reasoning_param = reasoning.LMReasoning("reasoning", [IdentityOption(), ZeroShotCoT(), PlanBefore()])
  few_shot_params = LMFewShot("few_shot", None, 8)

  inner_loop = InnerLoopBayesianOptimization(
    universal_params=[few_shot_params, reasoning_param], save_ckpt_interval=1
  )
  # outer_loop = OuterLoopOptimization(universal_params=[....])

  evaluator = EvaluatorPlugin(
    eval_set=train,
    n_parallel=20
  )

  # outer_loop.optimize(inner_loop, 
  #   n_trials=6, 
  #   script_path='/home/reyna/Cognify/examples/cognify_python_agent/workflow.py', 
  #   evaluator=evaluator,
  #   resource_ratio=1/3,
  #   log_dir=f'/home/reyna/Cognify/examples/cognify_python_agent/logs/logs_{uuid.uuid4()}')
  
  cost, pareto_frontier = inner_loop.optimize(
    script_path='/mnt/ssd4/lm_compiler/examples/cognify_python_agent/workflow.py',
    n_trials=9,
    evaluator=evaluator,
    log_dir=f'/mnt/ssd4/lm_compiler/examples/cognify_python_agent/opt_logs',
    throughput=3,
  )
  print(f"Optimization cost: {cost}")
  return pareto_frontier

def eval(data, config: InnerLoopBayesianOptimization.TrialLog):
  print(f"----- Testing trial {config.id} -----")
  trial, task = config.program
  print("  Params: {}".format(trial.params))
  f1, f2 = trial.values
  print("  Values: score= {}, cost= {}".format(f1, f2))
  
  evaluator = EvaluatorPlugin(
    eval_set=data,
    n_parallel=20,
  )
  eval_result = evaluator.evaluate(task)
  print(str(eval_result))

if __name__ == '__main__':
  # mp.set_start_method(method=None, force=True)
  mp.set_start_method('spawn')
  
  dataset = HumanEvalDataset()
  all_data = [(data["prompt"], data) for data in dataset.data]
  """
  Optimization
  """
  train, test = all_data[:40], all_data[40:]
  candidates = opt(train)
  
  """
  Eval
  """
  eval(test, candidates[3])
  
  """
  Dummy full evaluation
  """
  # eval_set = [(data["prompt"], data) for data in dataset.data]
  # evaluator = EvaluatorPlugin(
  #   eval_set=eval_set,
  #   n_parallel=20
  # )
  # task = EvalTask(
  #   script_path='/mnt/ssd4/lm_compiler/examples/cognify_python_agent/workflow.py',
  #   args=[],
  #   module_map_table=None,
  #   module_pool=None,
  # )
  # eval_result: EvaluationResult = evaluator.evaluate(task)
  # print(str(eval_result))