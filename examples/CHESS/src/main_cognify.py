# ⚠️ USE AT YOUR OWN RISK
# first: pip install pysqlite3-binary
# then in settings.py:

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import debugpy
import argparse
import json
from datetime import datetime

from runner.run_manager import RunManager
from runner.database_manager import DatabaseManager
from runner.logger import Logger
from pipeline.pipeline_manager import PipelineManager
from typing import Any, Dict, List, TypedDict, Callable
from langgraph.graph import END, StateGraph

from pipeline.keyword_extraction import keyword_extraction
from pipeline.entity_retrieval import entity_retrieval
from pipeline.context_retrieval import context_retrieval
from pipeline.column_filtering import column_filtering
from pipeline.table_selection import table_selection
from pipeline.column_selection import column_selection
from pipeline.candidate_generation import candidate_generation
from pipeline.revision import revision
from pipeline.evaluation import evaluation
from pipeline.annotated import cognify_registry
from compiler.IR.llm import LMConfig
from llm.models import get_llm_params

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
    return dataset[:10]


### Graph State ###
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


workflow = StateGraph(GraphState)
workflow.add_node("keyword_extraction", keyword_extraction)
workflow.add_node("entity_retrieval", entity_retrieval)
workflow.add_node("context_retrieval", context_retrieval)
workflow.add_node("column_filtering", column_filtering)
workflow.add_node("table_selection", table_selection)
workflow.add_node("column_selection", column_selection)
workflow.add_node("candidate_generation", candidate_generation)
workflow.add_node("revision", revision)
workflow.add_node("evaluation", evaluation)

workflow.set_entry_point("keyword_extraction")
workflow.add_edge("keyword_extraction", "entity_retrieval")
workflow.add_edge("entity_retrieval", "context_retrieval")
workflow.add_edge("context_retrieval", "column_filtering")
workflow.add_edge("column_filtering", "table_selection")
workflow.add_edge("table_selection", "column_selection")
workflow.add_edge("column_selection", "candidate_generation")
workflow.add_edge("candidate_generation", "revision")
workflow.add_edge("revision", "evaluation")
workflow.add_edge("evaluation", END)

app = workflow.compile()


def main():
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = parse_arguments()
    dataset = load_dataset(args.data_path)

    run_manager = RunManager(args)
    run_manager.initialize_tasks(dataset)
    
    # Set up the language model configurations
    pipeline_cfg = json.loads(args.pipeline_setup)
    for node_name, cfg in pipeline_cfg.items():
        if node_name in cognify_registry._cognify_lm_registry:
            lm = cognify_registry._cognify_lm_registry[node_name]
            engine_name = cfg["engine"]
            temperature = cfg.get("temperature", 0)
            base_uri = cfg.get("base_uri", None)
            
            lm_params = get_llm_params(engine=engine_name, temperature=temperature, base_uri=base_uri)
            model = lm_params.pop("model")
            lm.lm_config = LMConfig(
                provider='openai',
                model=model,
                kwargs=lm_params,
            )
    
    for task in run_manager.tasks:
        logger = Logger(db_id=task.db_id, question_id=task.question_id, result_directory=run_manager.result_directory)
        logger._set_log_level(args.log_level)
        logger.log(f"Processing task: {task.db_id} {task.question_id}", "info")
        database_manager = DatabaseManager(db_mode=args.data_mode, db_id=task.db_id)
        tentative_schema = database_manager.get_db_schema()
        pipeline_manager = PipelineManager(json.loads(args.pipeline_setup))
        initial_state = {
            "keys": {
                "task": task,
                "tentative_schema": tentative_schema,
                "execution_history": [],
            }
        }
        for state in app.stream(initial_state):
            continue
        run_manager.task_done((state['evaluation'], task.db_id, task.question_id))
    run_manager.generate_sql_files()


if __name__ == "__main__":
    # debugpy.listen(5678)
    # debugpy.wait_for_client()
    main()
