import argparse
import sys
import multiprocessing as mp
import os
import logging
import json
import debugpy

from cognify.optimizer.plugin import OptimizerSchema
from cognify.cognify_args import init_cognify_args, OptimizationArgs, EvaluationArgs, InspectionArgs
from cognify.optimizer.plugin import capture_module_from_fs
from cognify.optimizer.registry import get_registered_data_loader, get_registered_opt_score_fn
from cognify.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from cognify.optimizer.control_param import ControlParameter
from cognify.run.optimize import optimize
from cognify.run.evaluate import evaluate
from cognify._logging import _configure_logger

logger = logging.getLogger(__name__)


def from_cognify_args(args):
    if args.mode == 'optimize':
        return OptimizationArgs.from_cli_args(args)
    elif args.mode == 'evaluate':
        return EvaluationArgs.from_cli_args(args)
    elif args.mode == 'inspect':
        return InspectionArgs.from_cli_args(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    

def parse_pipeline_config_file(config_path):
    config_module = capture_module_from_fs(config_path)
    
    # load data
    data_loader_fn = get_registered_data_loader()
    train_set, val_set, test_set = data_loader_fn()
    logger.info(
        f"size of train set: {0 if not train_set else len(train_set)}, "
        f"val set: {0 if not val_set else len(val_set)}, "
        f"test set: {0 if not test_set else len(test_set)}"
    )
    
    # get optimizer control parameters
    control_param = ControlParameter.build_control_param(loaded_module=config_module)
    
    return (train_set, val_set, test_set), control_param

def optimize_routine(opt_args: OptimizationArgs):
    (train_set, val_set, test_set), control_param = parse_pipeline_config_file(opt_args.config)
    
    cost, frontier, opt_logs = optimize(
        script_path=opt_args.workflow,
        control_param=control_param,
        train_set=train_set,
        val_set=val_set,
        eval_fn=None,
        eval_path=opt_args.config,
    )
    return cost, frontier, opt_logs
    
    
def evaluate_routine(eval_args: EvaluationArgs):
    (train_set, val_set, test_set), control_param = parse_pipeline_config_file(eval_args.config)
    result = evaluate(
        control_param=control_param,
        config_id=eval_args.config_id,
        test_set=test_set,
        n_parallel=eval_args.n_parallel,
        eval_fn=None,
        eval_path=eval_args.config,
        save_to=eval_args.output_path,
    )
    return result

def inspect_routine(inspect_args: InspectionArgs):
    control_param = ControlParameter.build_control_param(inspect_args.control_param)
    
    # get dry run result on train set
    quality_constraint = None
    if control_param.quality_constraint is not None:
        dry_run_log_path = os.path.join(control_param.opt_history_log_dir, 'dry_run_train.json')
        if os.path.exists(dry_run_log_path):
            with open(dry_run_log_path, 'r') as f:
                dry_run_result = EvaluationResult.from_dict(json.load(f))
            logger.info(f"Loading existing dry run result at {dry_run_log_path}")
            quality_constraint = control_param.quality_constraint * dry_run_result.reduced_score
        else:
            logger.warning(f"Quality constraint is set but no dry run result found at {dry_run_log_path}, will ignore constraint")
            quality_constraint = None
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        quality_constraint=quality_constraint,
        save_config_to_file=False,
    )
    return opt_driver.inspect(inspect_args.dump_frontier_details)
    
    
def main():
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    parser = argparse.ArgumentParser()
    init_cognify_args(parser)
    raw_args = parser.parse_args()
    _configure_logger(raw_args.log_level)
    
    cognify_args = from_cognify_args(raw_args)
    if raw_args.mode == 'optimize':
        optimize_routine(cognify_args)
    elif raw_args.mode == 'evaluate':
        evaluate_routine(cognify_args)
    else:
        inspect_routine(cognify_args)
    return

if __name__ == "__main__":
    main()
    