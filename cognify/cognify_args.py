import argparse
import dataclasses
import logging
import os
import importlib.util
import json

logger = logging.getLogger(__name__)

@dataclasses.dataclass(kw_only=True)
class CommonArgs:
    workflow: str
    config: str = None
    log_level: str = 'INFO'
    
    def __post_init__(self):
        # Set missing values
        self.search_at_workflow_dir_if_not_set('config', 'py')
    
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            'workflow', 
            type=str,
            help="Path to the workflow script",
            metavar='path_to_workflow',
        )
        parser.add_argument(
            '-c', '--config',
            type=str,
            default=OptimizationArgs.config,
            help="Path to the configuration file for the optimization pipeline.\n"
            "If not provided, will search config.py in the same directory as workflow script.\n"
            "The file should contains the evaluator, data_loader, and optimizer settings.",
            metavar='path_to_config',
        )
        parser.add_argument(
            '-l', '--log_level',
            type=str,
            default=CommonArgs.log_level,
            help="Log level",
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            metavar='log_level',
        )
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'CommonArgs':
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def search_at_workflow_dir_if_not_set(self, attr, suffix):
        if getattr(self, attr) is None:
            workflow_dir = os.path.dirname(self.workflow)
            setattr(self, attr, os.path.join(workflow_dir, f'{attr}.{suffix}'))
        
@dataclasses.dataclass
class OptimizationArgs(CommonArgs):
    ...
    

@dataclasses.dataclass
class EvaluationArgs(CommonArgs):
    config_id: str
    n_parallel: int = 10
    output_path: str = None
    
    def __post_init__(self):
        super().__post_init__()
        self.search_at_workflow_dir_if_not_set('output_path', 'json')
    
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        CommonArgs.add_cli_args(parser)
        parser.add_argument(
            '-i', '--config_id', 
            type=str, 
            required=True, 
            help="Configuration ID for evaluation",
            metavar='config_id',
        )
        parser.add_argument(
            '-j', '--n_parallel', 
            type=int,
            default=EvaluationArgs.n_parallel,
            help="Number of parallel executions for evaluation. Please be aware of the rate limit of your model API provider.",
            metavar='n_parallel',
        )
        parser.add_argument(
            '-o', '--output_path', 
            type=str,
            default=EvaluationArgs.output_path,
            help="Path to the log file for evaluation results.\nIf not provided, will save to the same directory as workflow script.",
            metavar='path_to_output_json',
        )
        
@dataclasses.dataclass
class InspectionArgs(CommonArgs):
    dump_frontier_details: bool = False
    
    @staticmethod
    def add_cli_args(parser):
        CommonArgs.add_cli_args(parser)
        parser.add_argument(
            '-f', '--dump_frontier_details',
            action='store_true',
            help="Dump descriptive optimization details of all Pareto frontiers.",
        )
        
def init_cognify_args(parser):
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    
    opt_parser = subparsers.add_parser('optimize', formatter_class=argparse.RawTextHelpFormatter)
    OptimizationArgs.add_cli_args(opt_parser)
    
    eval_parser = subparsers.add_parser('evaluate', formatter_class=argparse.RawTextHelpFormatter)
    EvaluationArgs.add_cli_args(eval_parser)
    
    inspect_parser = subparsers.add_parser('inspect', formatter_class=argparse.RawTextHelpFormatter)
    InspectionArgs.add_cli_args(inspect_parser)