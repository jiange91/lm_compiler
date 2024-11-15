from typing import Union, Optional, Any, Tuple
import copy
import logging
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import joblib
from pathlib import Path
from queue import Queue
from collections import defaultdict


from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import LMConfig, LLMPredictor
from compiler.optimizer.utils import convert_to_comparable_repr, StateManager, StateManager_v2, OptionProfiler, PropagationEvaluator, ScorePath, DecisionNode

logger = logging.getLogger(__name__)

class BootStrap:
    """
    BootStrap class is used to bootstrap the optimizer 
    with a comprehensive set of different configurations and their intermediate results
    """
    def __init__(self):
        pass

    def bootstrap(self):
        pass


class BootStrapLMSelection(BootStrap):
    """
    If ground truth in trainset pairs are None,
    the optimizer is trained using teacher model's labels as ground truth
    """
    def __init__(
        self,
        workflow: Workflow,
        teachers: Union[dict[str, str], str],
        module_2_options: Union[dict[str, list[str]], list[str]],
        module_2_metric: Union[dict[str, callable], callable],
        final_output_metric: callable,
        trainset_input: list[StatePool],
        trainset_label: Optional[list[Any]] = None,
        max_sample_to_keep: int = 4,
    ):
        super().__init__()
        if not isinstance(teachers, dict):
            teachers = {m.name: teachers for m in workflow.modules}
        if not isinstance(module_2_options, dict):
            module_2_options = {m.name: module_2_options for m in workflow.modules}
        if not isinstance(module_2_metric, dict):
            module_2_metric = {m.name: module_2_metric for m in workflow.modules}
            
        self.workflow = workflow
        self.module_2_options = module_2_options
        # metric should be invoked with (gold, pred)
        self.module_2_metric = module_2_metric
        
        self.trainset_input = trainset_input
        self.trainset_label = trainset_label or [] # candidate always compare with this to estimatte the quality
        self.use_teacher_as_ground_truth = trainset_label is None
        self.teacher_final_outputs = []
        self.final_output_metric = final_output_metric
        self.final_output_scores = []
        
        self.teachers = teachers
        # The maximum number of output with different qualities to keep after each module
        # This param is per traning-input
        self.max_sample_to_keep = max_sample_to_keep
        self.sorted_target_modules: list[LLMPredictor] = self.workflow.sort(lambda x: isinstance(x, LLMPredictor))
        self.module_2_input_quality = {}
        self.score_tree = None

        # NOTE: Forward preceding non-LLM modules
        for state in self.trainset_input:
            self.workflow.run(state=state, stop_before=self.sorted_target_modules[0])
    
    def get_labels(self, label_path: str):
        if os.path.exists(label_path):
            logger.info(f"Loading labels from {label_path}")
            with open(label_path, 'r') as f:
                labels = json.load(f)
            # TODO: add more assertio to check label integrity
            assert len(labels) == len(self.trainset_input), "Each task should have a ground truth"
            
            for label in labels:
                self.teacher_final_outputs.append(label['final_output'])
                if self.use_teacher_as_ground_truth:
                    self.trainset_label.append(label['final_output'])
                    self.final_output_scores.append(1.0)
                else:
                    self.final_output_scores.append(label['final_score']['final_output'])
        else:
            # Get labels using teacher model
            for lm in self.sorted_target_modules:
                lm.lm_config['model'] = self.teachers[lm.name]
            labels = [] # idx -> {lm_name, output}
            # Get labels for each task
            output_labels = defaultdict(dict)
            for i, state in enumerate(self.trainset_input):
                logger.info(f'Generating labels for task: {i} ...')
                state_cpy = copy.deepcopy(state)
                self.workflow.reset()
                for module_idx, lm in enumerate(self.sorted_target_modules):
                    output_labels[lm.name] = {}
                    next_lm = self.sorted_target_modules[module_idx + 1] if module_idx + 1 < len(self.sorted_target_modules) else None
                    pred = self.workflow.run(state=state_cpy, start_from=lm, stop_before=next_lm)
                    for k, v in pred.items():
                        output_labels[lm.name][k] = copy.deepcopy(convert_to_comparable_repr(v))
                final_output = self.workflow.exit_result
                output_labels['final_output'] = convert_to_comparable_repr(final_output[self.workflow.exit_point[1]])
                self.teacher_final_outputs.append(output_labels['final_output'])
                if self.use_teacher_as_ground_truth:
                    self.trainset_label.append(output_labels['final_output'])
                output_labels['final_score'] = self.final_output_metric(
                    {'final_output': self.trainset_label[i]}, 
                    {'final_output': output_labels['final_output']},
                    state_cpy.all_news()
                )
                self.final_output_scores.append(output_labels['final_score']['final_output'])
                labels.append(output_labels)
            with open(label_path, 'w+') as f:
                json.dump(labels, f, indent=4)
            logger.info("Finish Generating labels")
        return labels

    def bootstrap(
        self,
        labels: list,
        profile_path: str,
    ):
        if os.path.exists(profile_path):
            logger.info(f"Loading profile from {profile_path}")
            with open(profile_path, 'r') as f:
                module_bootstrap_profile = json.load(f)
            return module_bootstrap_profile

        state_manager = StateManager_v2(self.trainset_input)
        module_bootstrap_profile = {}
        for module_idx, lm in enumerate(self.sorted_target_modules):
            logger.info(f"Bootstraping Module: {lm.name}")
            
            # forward until the next LM
            next_lm = self.sorted_target_modules[module_idx + 1] if module_idx + 1 < len(self.sorted_target_modules) else None
            
            # Get batch of input state for each task
            state_score_for_module = state_manager.prepare_state(lm.input_fields, self.max_sample_to_keep)
            new_state_scores: list[list[StateManager_v2.StateScoreType]] = [
                [] for _ in state_score_for_module
            ]
            new_profile_record = [
                [] for _ in self.module_2_options[lm.name]
            ]
            
            # Profile information propagation for each task
            for task_idx, task in enumerate(state_score_for_module):
                logger.info(f"Performing Task: {task_idx} ...")
                gold = labels[task_idx][lm.name]
                # State only contains input fields
                state_score_list: list[list[StateManager_v2.StateScoreType]] = task
                for option_idx, option in enumerate(self.module_2_options[lm.name]):
                    def option_runner(state: StatePool):
                        # run workflow
                        self.workflow.reset()
                        pred = self.workflow.run(state=state, start_from=lm, stop_before=next_lm)
                        comparable_pred = {k: convert_to_comparable_repr(v) for k, v in pred.items()}
                        output_quality = self.module_2_metric[lm.name](gold, comparable_pred)
                        # get final output quality if this is exit point
                        if lm.name == self.workflow.exit_point[0].name:
                            final_ground_truth = {'final_output': self.trainset_label[task_idx]}
                            final_quality = self.final_output_metric(
                                final_ground_truth, 
                                {'final_output': comparable_pred[self.workflow.exit_point[1]]},
                                state.all_news()
                            )
                            output_quality.update(final_quality)
                        return pred, output_quality
                    
                    option_profiler = OptionProfiler(option, state_score_list)
                    lm.lm_config['model'] = option
                    option_profiler.profie(option_runner)
                    # NOTE: state is grouped by task
                    new_state_scores[task_idx].extend(option_profiler.new_state_score)
                    # NOTE: profile result is grouped by option
                    new_profile_record[option_idx].extend(option_profiler.profile_record)
            state_manager.update_state(new_state_scores)
            module_bootstrap_profile[lm.name] = new_profile_record
        json.dump(module_bootstrap_profile, open(profile_path, 'w+'), indent=4)
        return module_bootstrap_profile
    
    def fit_curve_from_profile(self, module_bootstrap_profile, curve_path):
        curve = PropagationEvaluator(module_bootstrap_profile)
        
        if os.path.exists(curve_path):
            logger.info(f"Loading curve from {curve_path}")
            curve.load(curve_path)
            return curve
        
        curve.train(self.sorted_target_modules, self.module_2_options)
        curve.dump(curve_path)
        return curve
                        
    def search(self, prop_eval: PropagationEvaluator, gap, solution_path):
        if os.path.exists(solution_path):
            raise ValueError("Solution already exists")
        score_paths = ScorePath(self.sorted_target_modules, prop_eval.module_2_option_2_predictor)
        score_paths.build_tree(list(self.trainset_input[0].state.keys()))
        mean_target_final_score = np.mean(self.final_output_scores)
        solutions = []
        for p in score_paths.paths:
            perf_retain = p.state_scores['final_output'] / mean_target_final_score
            if perf_retain >= gap:
                solutions.append({'selections': p.selections, 'state_scores': p.state_scores, 'perf_retain': perf_retain})
        with open(solution_path, 'w+') as f:
            json.dump(solutions, f, indent=4)
        return solutions
        
    
    def compile(
        self,
        log_dir: str = 'bootstrap_log',
        gap: float = .95,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        """
        for all modules:
            for all options:
                for all input states:
                    generate_output
            filter_output
            output as input for next module
        """
        # Get ground truth labels
        labels = self.get_labels(os.path.join(log_dir, 'labels.json'))
        
        # Get profile for each module
        module_bootstrap_profile = self.bootstrap(labels, os.path.join(log_dir, 'module_option_profile.json'))
        
        # build curve for each option at each module from the profile
        prop_evaluator = self.fit_curve_from_profile(module_bootstrap_profile, os.path.join(log_dir, 'rag_curve.joblib'))
        
        # Build score tree and search for the best option
        config_candidates = self.search(prop_evaluator, gap, os.path.join(log_dir, 'solutions.json'))
        
        # Log token usage
        self.workflow.log_token_usage(os.path.join(log_dir, 'compile_token_usage.json'))
        
        return config_candidates