import dataclasses
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class OptimizationArgs:
    script_path: str
    control_param_path: str
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument('--script_path', type=str, required=True)
        parser.add_argument('--control_param_path', type=str, required=True)
    
    @classmethod
    def from_cli_args(cls, args):
        return cls(
            script_path=args.script_path,
            control_param_path=args.control_param_path,
        )

@dataclasses.dataclass
class EvaluationArgs:
    script_path: str
    config_id: str
    config_log_path: str
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument('--script_path', type=str, required=True)
        parser.add_argument('--config_id', type=str, required=True)
        parser.add_argument('--config_log_path', type=str, required=True)
    
    @classmethod
    def from_cli_args(cls, args):
        return cls(
            script_path=args.script_path,
            config_id=args.config_id,
            config_log_path=args.config_log_path,
        )
        
def init_cognify_args(parser):
    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True
    
    opt_parser = subparsers.add_parser('optimize')
    OptimizationArgs.add_cli_args(opt_parser)
    
    eval_parser = subparsers.add_parser('evaluate')
    EvaluationArgs.add_cli_args(eval_parser)
    
