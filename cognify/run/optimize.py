import os
import json
from typing import Optional, Union, Sequence, Callable
import logging

from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluation.evaluator import EvaluatorPlugin, EvalTask, EvaluationResult


logger = logging.getLogger(__name__)

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

def optimize(
    script_path: str,
    control_param: ControlParameter,
    train_set,
    *,
    eval_fn: Callable = None,
    eval_path: str = None,
    val_set = None,
):
    # Validate and prepare the pipeline
    assert eval_fn is not None or eval_path is not None, "Either eval_fn or eval_path should be provided"
    
    # create directory for logging
    if not os.path.exists(control_param.opt_history_log_dir):
        os.makedirs(control_param.opt_history_log_dir, exist_ok=True)
    
    # create evaluator
    evaluator = EvaluatorPlugin(
        trainset=train_set,
        evalset=val_set,
        testset=None,
        evaluator_path=eval_path,
        evaluator_fn=eval_fn,
        n_parallel=control_param.evaluator_batch_size,
    )

    # dry run on train set
    raw_result = dry_run(
        script_path=script_path, 
        evaluator=evaluator,
        log_dir=control_param.opt_history_log_dir,
    )
    
    # downsample data
    if control_param.train_down_sample > 0:
        downsample_data(
            script_path=script_path,
            source=evaluator,
            mode='train',
            sample_size=control_param.train_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
    if control_param.val_down_sample > 0:
        downsample_data(
            script_path=script_path,
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
        script_path=script_path,
    )
    return cost, pareto_frontier, opt_logs