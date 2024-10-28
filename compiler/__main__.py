import argparse
import sys
import multiprocessing as mp
import os
import logging
import importlib.util
import dataclasses
import json
from typing import Optional
import debugpy

from compiler.optimizer.plugin import OptimizerSchema
from compiler.cognify_args import init_cognify_args, OptimizationArgs, EvaluationArgs, InspectionArgs
from compiler.optimizer.plugin import capture_data_loader
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.core.driver import LayerConfig
from compiler.optimizer.core import driver, flow
from compiler.optimizer.core.unified_layer_opt import BottomLevelOptimization

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ControlParameter:
    opt_layer_configs: list[LayerConfig]
    opt_history_log_dir: str
    quality_constraint: float = 1.0
    train_down_sample: int = 0
    val_down_sample: int = 0
    evaluator_parallel: int = 20
    
    @classmethod
    def from_python_profile(cls, param_path):
        if not os.path.isfile(param_path):
            raise FileNotFoundError(f"The control param file {param_path} does not exist.")
        module_name = os.path.basename(param_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, param_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        attrs = [attr.name for attr in dataclasses.fields(cls)] 
        return cls(**{attr: getattr(module, attr) for attr in attrs})
    
    @classmethod
    def from_json_profile(cls, param_path):
        if not os.path.isfile(param_path):
            raise FileNotFoundError(f"The control param file {param_path} does not exist.")
        with open(param_path, 'r') as f:
            param_dict = json.load(f)
        attrs = [attr.name for attr in dataclasses.fields(cls)
                 if attr.name not in ['opt_layer_configs']]
        opt_layer_configs = [
            LayerConfig.from_dict(layer_dict) for layer_dict in param_dict['opt_layer_configs']
        ]
        return cls(
            opt_layer_configs=opt_layer_configs,
            **{attr: param_dict[attr] for attr in attrs},
        )
    
    def validate(self):
        # create directory for logging
        if not os.path.exists(self.opt_history_log_dir):
            os.makedirs(self.opt_history_log_dir, exist_ok=True)
        
    @classmethod
    def build_control_param(cls, control_param_path):
        if control_param_path.endswith('.py'):
            control_param = ControlParameter.from_python_profile(control_param_path)
        else:
            control_param = ControlParameter.from_json_profile(control_param_path)
        control_param.validate()
        return control_param

def from_cognify_args(args):
    if args.mode == 'optimize':
        return OptimizationArgs.from_cli_args(args)
    elif args.mode == 'evaluate':
        return EvaluationArgs.from_cli_args(args)
    elif args.mode == 'inspect':
        return InspectionArgs.from_cli_args(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
def dry_run(script_path, train_data, eval_parallel, log_dir):
    evaluator = EvaluatorPlugin(
        trainset=train_data,
        evalset=None,
        testset=None,
        n_parallel=eval_parallel,
    )
    eval_task = EvalTask(
        script_path=script_path,
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    logger.info(f"Dry run on train set: {len(train_data)} samples for optimizer analysis")
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

def downsample_data(script_path, source, mode, sample_size, log_dir):
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

def load_data(data_loader_path):
    data_loader_fn = capture_data_loader(data_loader_path)
    train_set, val_set, test_set = data_loader_fn()
    logger.info(f"size of train set: {len(train_set)}, val set: {len(val_set)}, test set: {len(test_set)}")
    return train_set, val_set, test_set
    

def optimize_routine(opt_args: OptimizationArgs):
    # load data
    train_set, val_set, test_set = load_data(opt_args.data_loader_path)
    
    # get optimizer control parameters
    control_param = ControlParameter.build_control_param(opt_args.control_param_path)
    
    # create evaluator
    evaluator = EvaluatorPlugin(
        trainset=train_set,
        evalset=val_set,
        testset=test_set,
        n_parallel=control_param.evaluator_parallel,
    )

    # dry run on train set
    raw_result = dry_run(
        script_path=opt_args.script_path, 
        train_data=train_set, 
        eval_parallel=control_param.evaluator_parallel,
        log_dir=control_param.opt_history_log_dir,
    )
    
    # downsample data
    if control_param.train_down_sample > 0:
        downsample_data(
            script_path=opt_args.script_path,
            source=evaluator,
            mode='train',
            sample_size=control_param.train_down_sample,
            log_dir=control_param.opt_history_log_dir,
        )
    if control_param.val_down_sample > 0:
        downsample_data(
            script_path=opt_args.script_path,
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
        script_path=opt_args.script_path,
    )
    return
    
    
def evaluate_routine(eval_args: EvaluationArgs):
    _, _, test_set = load_data(eval_args.data_loader_path)
    
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=test_set,
        n_parallel=eval_args.n_parallel,
    )
    result = BottomLevelOptimization.easy_eval(
        evaluator=evaluator,
        config_id=eval_args.config_id,
        opt_log_path=eval_args.config_log_path,
    )
    logger.info(result)
    if eval_args.log_path is not None:
        with open(eval_args.log_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=4)
    return result

def inspect_routine(inspect_args: InspectionArgs):
    control_param = ControlParameter.build_control_param(inspect_args.control_param_path)
    
    # get dry run result on train set
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
    return opt_driver.inspect()
    
    
def main():
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    parser = argparse.ArgumentParser()
    init_cognify_args(parser)
    raw_args = parser.parse_args()
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
    