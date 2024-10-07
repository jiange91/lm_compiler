import json
from code_completion_agent import code_completion_agent
from code_finalize_agent import code_finalize_agent
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn
from humaneval.humaneval import HumanEvalDataset, check_correctness_thread, check_correctness
from compiler.optimizer.evaluation.evaluator import EvaluatorInterface, EvaluationResult, EvaluatorPlugin, EvalTask

from compiler.utils import load_api_key
import string

load_api_key('/home/reyna/Cognify/secrets.toml')

@register_opt_program_entry
def mainworkflow(incomplete_code):
  completed_code = code_completion_agent.invoke({"incomplete_code": incomplete_code}).content
  finalized_code = code_finalize_agent.invoke({"completed_code": completed_code, "incomplete_code": incomplete_code}).content
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


if __name__ == '__main__':
  dataset = HumanEvalDataset()
  total = len(dataset.data)
  correct = 0
  
  """
  Simple test
  """
  for i in range(164):
    problem = dataset.data[i]
    code = mainworkflow(dataset.data[i]["prompt"])
    score = score_fn(problem, code)
    correct += score
    print(f"Test {i+1}: {score}")
  
  # def process_datum(i, datum):
  #   mainworkflow(datum["task_id"], datum["prompt"])
  #   return i, score_fn(dataset, i)

  # with ThreadPoolExecutor() as executor:
    # futures = [executor.submit(process_datum, i, datum) for i, datum in enumerate(dataset.data) if i in unprocessed_set]
  
  # error_no_pass = 0
  # for i in range(total):
  #   passed = score_fn(dataset, i)
  #   correct += passed
  print(f"Pass@1: {correct}/{164}, ({correct/total*100:.2f}%)")
