import argparse
import dataclasses
import logging
import os
import importlib.util
import json

logger = logging.getLogger(__name__)

@dataclasses.dataclass(kw_only=True)
class CommonArgs:
    script_path: str
    data_loader_path: str
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument('--script_path', type=str, required=True)
        parser.add_argument('--data_loader_path', type=str, required=True)
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

@dataclasses.dataclass
class OptimizationArgs(CommonArgs):
    control_param_path: str
    
    @staticmethod
    def add_cli_args(parser):
        CommonArgs.add_cli_args(parser)
        parser.add_argument('--control_param_path', type=str, required=True)

@dataclasses.dataclass
class EvaluationArgs(CommonArgs):
    config_id: str
    config_log_path: str
    n_parallel: int = 10
    log_path: str = None
    
    @staticmethod
    def add_cli_args(parser):
        CommonArgs.add_cli_args(parser)
        parser.add_argument('--config_id', type=str, required=True)
        parser.add_argument('--config_log_path', type=str, required=True)
        parser.add_argument('--n_parallel', type=int, default=EvaluationArgs.n_parallel)
        parser.add_argument('--log_path', type=str, required=False)
        
@dataclasses.dataclass
class InspectionArgs:
    control_param_path: str
    dump_frontier_details: bool = False
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument('--control_param_path', type=str, required=True)
        parser.add_argument('--dump_frontier_details', action='store_true')
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})
        
def init_cognify_args(parser):
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    
    opt_parser = subparsers.add_parser('optimize')
    OptimizationArgs.add_cli_args(opt_parser)
    
    eval_parser = subparsers.add_parser('evaluate')
    EvaluationArgs.add_cli_args(eval_parser)
    
    inspect_parser = subparsers.add_parser('inspect')
    InspectionArgs.add_cli_args(inspect_parser)

