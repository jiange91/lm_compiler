import os
import json
from typing import Optional, Union, Callable
import logging

from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluation.evaluator import (
    EvaluatorPlugin,
    EvaluationResult,
)
from cognify.optimizer.evaluation.metric import MetricBase


logger = logging.getLogger(__name__)


def evaluate(
    *,
    config_id: str,
    test_set,
    opt_result_path: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
    n_parallel: int = 10,
    eval_fn: Union[Callable, MetricBase] = None,
    eval_path: str = None,
    save_to: str = None,
) -> EvaluationResult:
    assert (
        control_param or opt_result_path
    ), "Either control_param or opt_result_path should be provided"
    # If both are provided, control_param will be used

    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, "control_param.json")
        control_param = ControlParameter.from_json_profile(control_param_save_path)

    if eval_fn is not None:
        if isinstance(eval_fn, MetricBase):
            eval_fn = eval_fn.score
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
    )
    result = opt_driver.evaluate(
        evaluator=evaluator,
        config_id=config_id,
    )

    if save_to is not None:
        with open(save_to, "w") as f:
            json.dump(result.to_dict(), f, indent=4)
    return result


def load_workflow(
    *,
    config_id: str,
    opt_result_path: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
) -> Callable:
    assert (
        control_param or opt_result_path
    ), "Either control_param or opt_result_path should be provided"
    # If both are provided, control_param will be used

    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, "control_param.json")
        control_param = ControlParameter.from_json_profile(control_param_save_path)

    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_setup.layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
    )
    schema, _ = opt_driver.load(config_id)
    return schema.program
