import os
import json
from typing import Union, Optional, Any, Tuple
import copy
import logging
import optunahub
import optuna
import numpy as np

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import LMConfig, LLMPredictor
from compiler.optimizer.utils import convert_to_comparable_repr, StateManager, StateManager_v2, OptionProfiler, PropagationEvaluator, ScorePath, DecisionNode
from compiler.utils import get_bill


logger = logging.getLogger(__name__)


class HOLMSelection:
    """
    we care about performance recovery, i.e. R = p(outs | gt) / p(teacher | gt) >= gap
    """
    def __init__(
        self,
        workflow: Workflow,
        teachers: Union[dict[str, str], str],
        module_2_options: Union[dict[str, list[str]], list[str]],
        final_output_metric: callable,
        trainset_input: list[StatePool],
        trainset_label: Optional[list[Any]] = None,
    ):
        if not isinstance(teachers, dict):
            teachers = {m.name: teachers for m in workflow.modules}
        if not isinstance(module_2_options, dict):
            module_2_options = {m.name: module_2_options for m in workflow.modules}
        self.workflow = workflow
        self.teachers = teachers
        self.module_2_options = module_2_options
        self.final_output_metric = final_output_metric
        self.trainset_input = trainset_input
        self.trainset_label = trainset_label
        self.sorted_target_modules: list[LLMPredictor] = self.workflow.sort(lambda x: isinstance(x, LLMPredictor))
        
        self.teacher_performance = []
        self.tpe_logs = {}

    def batch_run_and_eval(self, set_label = False):
        preds = []
        prices = []
        states = []
        for state in self.trainset_input:
            state_cpy = copy.deepcopy(state)
            self.workflow.reset_modules(True)
            pred = self.workflow.run(state_cpy)
            preds.append(pred)
            
            self.workflow.update_token_usage_summary()
            price = get_bill(self.workflow.token_usage_buffer)[0]
            prices.append(price)
            
            states.append(state_cpy)
        if set_label:
            self.trainset_label = preds
        scores = []
        for pred, gt in zip(preds, self.trainset_label):
            scores.append(self.final_output_metric(gt, pred)[self.workflow.exit_point[1]])
        return preds, scores, prices
        
    def get_teacher_performance(self, pred_path: str):
        if os.path.exists(pred_path):
            logger.info(f"Loading labels from {pred_path}")
            with open(pred_path, 'r') as f:
                teacher_trials = json.load(f)
            for trial, gt in zip(teacher_trials, self.trainset_label):
                self.teacher_performance.append(self.final_output_metric(gt, trial['pred'])[self.workflow.exit_point[1]])
            logger.info(f"average price: {np.mean([t['price'] for t in teacher_trials])}")
        else:
            for lm in self.sorted_target_modules:
                lm.lm_config['model'] = self.teachers[lm.name]
            teacher_preds, scores, prices = self.batch_run_and_eval(set_label=True)
            teacher_trials = [{'pred': pred, 'price': price} for pred, price in zip(teacher_preds, prices)]
            json.dump(teacher_trials, open(pred_path, 'w+'), indent=4)
            self.teacher_performance = scores
    
    def get_objective_function(self):
        def objective_function(trial):
            logging.info(f"Trial: {trial.number}")
            self.tpe_logs[trial.number] = {}
            for lm in self.sorted_target_modules:
                lm_options = self.module_2_options[lm.name]
                selected = trial.suggest_categorical(lm.name, lm_options)
                lm.lm_config['model'] = selected
                self.tpe_logs[trial.number][lm.name] = selected
            preds, scores, prices = self.batch_run_and_eval()
            # take the average of the performance recovery across different tasks
            avg_performance_recovery = sum(ours / teacher for ours, teacher in zip(scores, self.teacher_performance)) / len(scores)
            # Get the avg price as second objective
            avg_price = sum(prices) / len(prices)
            self.tpe_logs[trial.number]['performance_recovery'] = avg_performance_recovery
            self.tpe_logs[trial.number]['price'] = avg_price
            return avg_performance_recovery, avg_price
        
        return objective_function
    
    def compile(
        self,
        log_dir: str = 'holm_log',
        gap: float = .95,
        prior_trials: list = None,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.get_teacher_performance(os.path.join(log_dir, 'teacher_trials.json'))
        
        obj_func = self.get_objective_function()
        study = optuna.create_study(directions=['maximize', 'minimize'])
        if prior_trials:
            for trial in prior_trials:
                study.enqueue_trial(trial)
        study.optimize(obj_func, n_trials=10)
        
        json.dump(self.tpe_logs, open(os.path.join(log_dir, 'tpe_logs.json'), 'w+'), indent=4)
        
        for i, best_trial in enumerate(study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, best_trial.number))
            print("  Params: {}".format(best_trial.params))
            f1, f2 = best_trial.values
            print("  Values: f1={}, f2={}".format(f1, f2))
            