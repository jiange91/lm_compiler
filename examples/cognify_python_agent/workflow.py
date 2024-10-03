import json
from code_completion_agent import code_completion_agent, code_completion_transit_agent
from code_finalize_agent import code_finalize_agent, code_finalize_transit_agent
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn
from humaneval.humaneval import HumanEvalDataset, check_correctness_thread, check_correctness
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask

from compiler.utils import load_api_key
import string

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

@register_opt_program_entry
def mainworkflow(incomplete_code):
  completed_code = None
  for i in range(10):
    incomplete_code = code_completion_agent.invoke({"incomplete_code": incomplete_code}).content
    output_node = code_completion_transit_agent.invoke({"completed_code": incomplete_code}).content
    completed_code = incomplete_code
    if '<node>Receive_Incomplete_Code</node>' in output_node:
      continue
    elif '<node>Finalize_Code</node>' in output_node:
      break
    else:
      raise Exception(f"Invalid output node by code completion transit agent: {output_node}", output_node)
  
  finalized_code = None
  for i in range(10):
    completed_code = code_finalize_agent.invoke({"completed_code": completed_code}).content
    output_node = code_finalize_transit_agent.invoke({"completed_code": completed_code}).content
    finalized_code = completed_code
    if '<node>Finalize_Code</node>' in output_node:
      continue
    elif '<node>end_node</node>' in output_node:
      break
    else:
      raise Exception(f"Invalid output node by code finalize transit agent: {output_node}", output_node)
  return finalized_code

@register_opt_score_fn
def score_fn(problem: str, pred: str):
  split_completion = pred.split('\n')
  parsed_lines = []
  for line in split_completion:
    if "<result>" in line or "</result>" in line or "```" in line or "python" in line:
      continue
    parsed_lines.append(line)
  completion = '\n'.join(parsed_lines)

  result = check_correctness_thread(problem, completion, timeout=3.0)
  return 1.0 if result["passed"] else 0.0
  raise Exception("Label not found in dataset")


if __name__ == '__main__':
  dataset = HumanEvalDataset()
  total = len(dataset.data)
  correct = 0
  
  """
  Simple test
  """
  problem = dataset.data[0]
  code = mainworkflow(dataset.data[0]["prompt"])
  score = score_fn(problem, code)
  print(score)
  
  # def process_datum(i, datum):
  #   mainworkflow(datum["task_id"], datum["prompt"])
  #   return i, score_fn(dataset, i)

  # with ThreadPoolExecutor() as executor:
    # futures = [executor.submit(process_datum, i, datum) for i, datum in enumerate(dataset.data) if i in unprocessed_set]
  
  # error_no_pass = 0
  # for i in range(total):
  #   passed = score_fn(dataset, i)
  #   correct += passed
  # print(f"Pass@1: {correct}/{total}, ({correct/total*100:.2f}%)")
