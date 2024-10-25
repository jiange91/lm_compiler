import argparse
import sys
import multiprocessing as mp
import os

from compiler.optimizer.plugin import OptimizerSchema
from compiler.optimizer.cognify import init_cognify_args, OptimizationArgs, EvaluationArgs

def from_cognify_args(args):
    if args.mode == 'optimize':
        return OptimizationArgs.from_cli_args(args)
    elif args.mode == 'evaluate':
        return EvaluationArgs.from_cli_args(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

def optimize_routine(opt_args: OptimizationArgs):
    ...
    
def evaluate_routine(eval_args: EvaluationArgs):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_cognify_args(parser)
    raw_args = parser.parse_args()
    cognify_args = from_cognify_args(raw_args)
    if raw_args.mode == 'optimize':
        optimize_routine(cognify_args)
    else:
        evaluate_routine(cognify_args)
    