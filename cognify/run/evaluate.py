import os
import json
from typing import Optional, Union, Sequence, Callable
import logging

from cognify.optimizer.plugin import OptimizerSchema
from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluation.evaluator import EvaluatorPlugin, EvalTask, EvaluationResult


logger = logging.getLogger(__name__)

def evaluate(
    control_param: ControlParameter,
    config_id: str,
    test_set,
    *,
    n_parallel: int = 10,
    eval_fn: Callable = None,
    eval_path: str = None,
    save_to: str = None,
) -> EvaluationResult:
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=test_set,
        evaluator_path=eval_path,
        evaluator_fn=eval_fn,
        n_parallel=n_parallel,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_setup.layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        save_config_to_file=False,
    )
    result = opt_driver.evaluate(
        evaluator=evaluator,
        config_id=config_id,
    )
    
    if save_to is not None:
        with open(save_to, 'w') as f:
            json.dump(result.to_dict(), f, indent=4)
    return result

def load_workflow(
    control_param: ControlParameter,
    config_id: str,
) -> Callable:
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_setup.layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
        save_config_to_file=False,
    )
    schema, _ = opt_driver.load(config_id)
    return schema.program