import json
from code_completion_agent import code_completion_agent
from code_finalize_agent import code_finalize_agent
from cognify.optimizer import register_opt_workflow, register_opt_score_fn
from humaneval.humaneval import HumanEvalDataset, check_correctness_thread, check_correctness
from cognify.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask

from cognify.utils import load_api_key
import string

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

@register_opt_workflow
def mainworkflow(incomplete_code):
  completed_code = code_completion_agent(inputs={"incomplete_function": incomplete_code}).choices[0].message.content
  finalized_code = code_finalize_agent(inputs={"incomplete_function": incomplete_code, "completed_code": completed_code}).choices[0].message.content
  return finalized_code

@register_opt_score_fn
def score_fn(problem, pred: str):
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
  size = 2
  for i in range(size):
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
  print(f"Pass@1: {correct}/{size}, ({correct/size*100:.2f}%)")
