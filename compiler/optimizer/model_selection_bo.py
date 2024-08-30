import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable
import copy
import logging
import optunahub
import optuna
import numpy as np

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor
from compiler.utils import get_bill


logger = logging.getLogger(__name__)


class LMSelectionBayesianOptimization:
    """
    we care about performance recovery, i.e. R = perf(outs | gt) / perf(teacher | gt) >= gap
    """
    def __init__(
        self,
        workflow: Workflow,
        teachers: Union[dict[str, str], str],
        module_2_options: Union[dict[str, list[str]], list[str]],
        final_output_metric: Callable[[Any, StatePool], Any],
        trainset_input: Iterable[StatePool],
        trainset_label: Iterable[Any],
    ):
        self.lm_modules : list[LLMPredictor] = workflow.get_all_modules(lambda x: isinstance(x, LLMPredictor))
        if not isinstance(teachers, dict):
            teachers = {m: teachers for m in self.lm_modules}
        if not isinstance(module_2_options, dict):
            module_2_options = {m: module_2_options for m in self.lm_modules}
        self.workflow = workflow
        self.teachers = teachers
        self.module_2_options = module_2_options
        self.final_output_metric = final_output_metric
        self.trainset_input = trainset_input
        self.trainset_label = trainset_label
        
        self.teacher_performance = []
        self.tpe_logs = {}
        self.tpe_distributions = {}
        for lm in self.lm_modules:
            self.tpe_distributions[lm.name] = optuna.distributions.CategoricalDistribution(module_2_options[lm.name])

    def batch_run_and_eval(self):
        prices = []
        states = []
        for state in self.trainset_input:
            state_cpy = copy.deepcopy(state)
            self.workflow.reset(True)
            self.workflow.pregel_run(state_cpy)
            
            self.workflow.update_token_usage_summary()
            price = get_bill(self.workflow.token_usage_buffer)[0]
            prices.append(price)
            
            states.append(state_cpy)
        scores = []
        for gt, state in zip(self.trainset_label, states):
            scores.append(self.final_output_metric(gt, state))
        return states, scores, prices
        
    def get_teacher_performance(self, pred_path: str):
        if os.path.exists(pred_path):
            logger.info(f"Loading metrics and prices from {pred_path}")
            with open(pred_path, 'r') as f:
                teacher_trials = json.load(f)
            for trial in teacher_trials:
                self.teacher_performance.append(trial['score'])
        else:
            for lm in self.lm_modules:
                lm.lm_config['model'] = self.teachers[lm.name]
            teacher_states, scores, prices = self.batch_run_and_eval()
            teacher_trials = [{'score': score, 'price': price} for score, price in zip(scores, prices)]
            json.dump(teacher_trials, open(pred_path, 'w+'), indent=4)
            self.teacher_performance = scores
        self.teacher_avg_price = np.mean([t['price'] for t in teacher_trials])
        logger.info(f"average teacher price: {self.teacher_avg_price}")
    
    def get_objective_function(self):
        def objective_function(trial):
            logging.info(f"Trial: {trial.number}")
            self.tpe_logs[trial.number] = {}
            self.tpe_logs[trial.number]['params'] = {}
            for lm in self.lm_modules:
                lm_options = self.module_2_options[lm.name]
                selected = trial.suggest_categorical(lm.name, lm_options)
                lm.lm_config['model'] = selected
                self.tpe_logs[trial.number]['params'][lm.name] = selected
            states, scores, prices = self.batch_run_and_eval()
            # take the average of the performance recovery across different tasks
            avg_performance_recovery = sum(ours / teacher for ours, teacher in zip(scores, self.teacher_performance)) / len(scores)
            # Get the avg price as second objective
            avg_price = sum(prices) / len(prices)
            self.tpe_logs[trial.number]['performance_recovery'] = avg_performance_recovery
            self.tpe_logs[trial.number]['price'] = avg_price
            return avg_performance_recovery, avg_price
        
        return objective_function

    def reduce_cold_start(self, base_model: str, important_lms: list[str], study: optuna.Study):
        base_config = {lm.name: base_model for lm in self.lm_modules}
        for lm_name in important_lms:
            for option in self.module_2_options[lm_name]:
                config = copy.deepcopy(base_config)
                config[lm_name] = option
                study.enqueue_trial(config)
        warm_start = len(study.get_trials(False))
        logger.info(f"Cold start reduction finished, enqueued: {warm_start} trials.")
        return warm_start
    
    def optimize(
        self,
        n_trials: int,
        log_dir: str = 'holm_log',
        gap: float = .95,
        base_model: str = None,
        important_lms: list[str] = None,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if important_lms is not None:
            assert base_model is not None, "base_model must be provided when important_lms is not None"

        obj_func = self.get_objective_function()
        study = optuna.create_study(directions=['maximize', 'minimize'])
        self.get_teacher_performance(os.path.join(log_dir, 'teacher_trials.json'))
        
        if os.path.exists(os.path.join(log_dir, 'tpe_logs.json')):
            with open(os.path.join(log_dir, 'tpe_logs.json'), 'r') as f:
                self.tpe_logs = json.load(f)
                for trial_id, meta in self.tpe_logs.items():
                    trial = optuna.trial.create_trial(
                        params=meta['params'],
                        values=[meta['performance_recovery'], meta['price']],
                        distributions=self.tpe_distributions,
                    )
                    study.add_trial(trial)
        else:
            warm_start = 0
            if important_lms is not None:
                warm_start = self.reduce_cold_start(base_model, important_lms, study)
            
            study.optimize(obj_func, n_trials=n_trials + warm_start)
            
            json.dump(self.tpe_logs, open(os.path.join(log_dir, 'tpe_logs.json'), 'w+'), indent=4)
        
        best_trial = None
        for i, trial in enumerate(study.best_trials):
            print("The {}-th Pareto solution was found at Trial#{}.".format(i, trial.number))
            print("  Params: {}".format(trial.params))
            f1, f2 = trial.values
            print("  Values: f1={}, f2={}".format(f1, f2))
            
            if f1 >= gap:
                if best_trial is None or f2 < best_trial.values[1]:
                    best_trial = trial
        
        print('-' * 40)
        if best_trial is not None:
            print(f"best config: {best_trial.params}")
            print(f"performance recovery: {best_trial.values[0]}")
            print(f"cost save by: {self.teacher_avg_price / best_trial.values[1]}x")
        else:
            print("No solution found.")
        print('-' * 40)