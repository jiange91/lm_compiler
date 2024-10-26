import argparse
import json
from datetime import datetime
import os
import debugpy
import multiprocessing as mp

from runner.task import Task
from typing import Any, Dict, List, TypedDict, Callable

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline with the specified configuration."
    )
    parser.add_argument(
        "--data_mode", type=str, required=True, help="Mode of the data to be processed."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file."
    )
    parser.add_argument(
        "--pipeline_nodes",
        type=str,
        required=True,
        help="Pipeline nodes configuration.",
    )
    parser.add_argument(
        "--pipeline_setup",
        type=str,
        required=True,
        help="Pipeline setup in JSON format.",
    )
    parser.add_argument(
        "--use_checkpoint", action="store_true", help="Flag to use checkpointing."
    )
    parser.add_argument(
        "--checkpoint_nodes",
        type=str,
        required=False,
        help="Checkpoint nodes configuration.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=False, help="Directory for checkpoints."
    )
    parser.add_argument(
        "--log_level", type=str, default="warning", help="Logging level."
    )
    args = parser.parse_args()

    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.use_checkpoint:
        print("Using checkpoint")
        if not args.checkpoint_nodes:
            raise ValueError("Please provide the checkpoint nodes to use checkpoint")
        args.checkpoint_nodes = args.checkpoint_nodes.split("+")
        if not args.checkpoint_dir:
            raise ValueError("Please provide the checkpoint path to use checkpoint")

    return args


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """
    Loads the dataset from the specified path.

    Args:
        data_path (str): Path to the data file.

    Returns:
        List[Dict[str, Any]]: The loaded dataset.
    """
    with open(data_path, "r") as file:
        dataset = json.load(file)
    return dataset[:]


if __name__ == "__main__":
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    
    args = parse_arguments()
    dataset = load_dataset(args.data_path)
    
    inputs = []
    dir_prefix = 'cognify_results/raw_test_all_cot_no_demos_generation_DC'
    for data in dataset:
        task = Task(data)
        result_dir = f"{dir_prefix}/{task.db_id}/{task.question_id}/{args.run_start_time}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        inputs.append(
            {
                'args': args,
                'dataset': [data],
                'result_directory': result_dir,
            }
        )
    eval_data = [(input, None) for input in inputs]
    
    from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask 
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/CHESS/src/cognify_worker.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=eval_data,
        n_parallel=40,
    )
    eval_result = evaluator.get_score('test', plain_task, show_process=True)
    print(eval_result)
    with open(f'{dir_prefix}/eval_result.json', 'w') as f:
        json.dump(eval_result.to_dict(), f, indent=4)