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
from compiler.optimizer import register_opt_program_entry, register_opt_score_fn


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


@register_opt_program_entry
def worker(input):
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = input['args']
    dataset = input['dataset']
    result_directory = input['result_directory']
    assert len(dataset) == 1, "Worker process perform one task at a time"

    run_manager = RunManager(args, result_directory)
    run_manager.initialize_tasks(dataset)
    task = run_manager.tasks[0]
    
    result = run_manager._app_worker(task, app)
    run_manager.task_done(result, show_progress=False) 

    return run_manager.statistics_manager.statistics.to_dict()

@register_opt_score_fn
def eval(label, stats):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in stats['counts'].values())
    return 1.0 if correct else 0.0
