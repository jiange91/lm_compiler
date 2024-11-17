import os
import logging
from typing import Optional
import json

from cognify.optimizer.control_param import ControlParameter
from cognify.cognify_args import InspectionArgs
from cognify.optimizer.evaluation.evaluator import EvaluationResult
from cognify.optimizer.core import driver

logger = logging.getLogger(__name__)

def inspect(
    opt_result_path: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
    dump_frontier_details: bool = False,
):
    assert control_param or opt_result_path, "Either control_param or opt_result_path should be provided"
    # If both are provided, control_param will be used
    
    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, 'control_param.json')
        control_param = ControlParameter.from_json_profile(control_param_save_path)
    
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
    )
    opt_driver.inspect(dump_frontier_details)