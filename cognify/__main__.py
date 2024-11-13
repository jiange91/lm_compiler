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
from cognify.optimizer.core import driver
from cognify.optimizer.control_param import ControlParameter
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
    
def dry_run(script_path, evaluator: EvaluatorPlugin, log_dir):
    eval_task = EvalTask(
        script_path=script_path,
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
        trace_back=["dry_run"],
    )
    logger.info(f"Dry run on train set: {len(evaluator.dataset['train'])} samples for optimizer analysis")
    dry_run_log_path = os.path.join(log_dir, 'dry_run_train.json')
    
    if os.path.exists(dry_run_log_path):
        with open(dry_run_log_path, 'r') as f:
            dry_run_result = EvaluationResult.from_dict(json.load(f))
        logger.info(f"Loading existing dry run result at {dry_run_log_path}")
        return dry_run_result
    
    result = evaluator.get_score('train', eval_task, show_process=True)
    with open(dry_run_log_path, 'w+') as f:
        json.dump(result.to_dict(), f, indent=4)
    logger.info(f"Dry run result saved to {dry_run_log_path}")
    return result

def downsample_data(script_path, source: EvaluatorPlugin, mode, sample_size, log_dir):
    plain_task = EvalTask(
        script_path=script_path,
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    source.down_sample(
        sample_size=sample_size,
        mode=mode,
        task=plain_task, 
        sample_mode='difficulty',
        log_dir=log_dir,
    )

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
    
    # create evaluator
    evaluator = EvaluatorPlugin(
        evaluator_path=opt_args.config,
        trainset=train_set,
        evalset=val_set,
        testset=test_set,
        n_parallel=control_param.evaluator_batch_size,
    )

    # dry run on train set
    raw_result = dry_run(
        script_path=opt_args.workflow, 
        evaluator=evaluator,
        log_dir=control_param.opt_history_log_dir,
    )
    
    # downsample data
    if control_param.train_down_sample > 0:
        downsample_data(
            script_path=opt_args.workflow,
            source=evaluator,
            mode='train',
            sample_size=control_param.train_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
    if control_param.val_down_sample > 0:
        downsample_data(
            script_path=opt_args.workflow,
            source=evaluator,
            mode='eval',
            sample_size=control_param.val_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
        
    # build optimizer from parameters
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        quality_constraint=control_param.quality_constraint * raw_result.reduced_score,
    )
    
    cost, pareto_frontier, opt_logs = opt_driver.run(
        evaluator=evaluator,
        script_path=opt_args.workflow,
    )
    return
    
    
def evaluate_routine(eval_args: EvaluationArgs):
    _, _, test_set = load_data(eval_args.data_loader)
    
    control_param = ControlParameter.build_control_param(eval_args.control_param)
    
    evaluator = EvaluatorPlugin(
        evaluator_path=eval_args.evaluator,
        trainset=None,
        evalset=None,
        testset=test_set,
        n_parallel=eval_args.n_parallel,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        save_config_to_file=False,
    )
    result = opt_driver.evaluate(
        evaluator=evaluator,
        config_id=eval_args.config_id,
    )
    logger.info(result)
    if eval_args.output_path is not None:
        with open(eval_args.output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=4)
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
    