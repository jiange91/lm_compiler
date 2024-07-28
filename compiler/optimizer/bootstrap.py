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
            for i, state in enumerate(self.trainset_input):
                logger.info(f'Generating labels for task: {i} ...')
                state_cpy = copy.deepcopy(state)
                self.workflow.reset_modules()
                final_output = self.workflow.run(state_cpy)
                output_labels = {
                    lm.name: {
                        k: copy.deepcopy(convert_to_comparable_repr(v))
                            for k, v in lm.outputs[-1].items()
                        }
                    for lm in self.sorted_target_modules
                }
                output_labels['final_output'] = convert_to_comparable_repr(final_output)
                self.teacher_final_outputs.append(output_labels['final_output'])
                if not self.use_teacher_as_ground_truth:
                    output_labels['final_score'] = self.final_output_metric(
                        {'final_output': self.trainset_label[i]}, 
                        {'final_output': output_labels['final_output']},
                    )
                    self.final_output_scores.append(output_labels['final_score'])
                else:
                    self.trainset_label.append(output_labels['final_output'])
                    self.final_output_scores.append(1.0)
                labels.append(output_labels)
            with open(label_path, 'w+') as f:
                json.dump(labels, f, indent=4)
            logger.info("Finish Generating labels")
        return labels

    def bootstrap(
        self,
        trainset: list[StatePool],
        labels: list,
        profile_path: str,
    ):
        if os.path.exists(profile_path):
            logger.info(f"Loading profile from {profile_path}")
            with open(profile_path, 'r') as f:
                module_bootstrap_profile = json.load(f)
            return module_bootstrap_profile
        
        entry_point = self.workflow.root
        dummy_input_quality = self.module_2_metric[entry_point.name](
            {'input': 'dummy'}, {'input': 'dummy'}
        )
        self.module_2_input_quality[self.workflow.root.name] = [[dummy_input_quality] for _ in trainset]
        
        module_bootstrap_profile = {}
        state_manager = StateManager(trainset)
        
        for module_idx, lm in enumerate(self.sorted_target_modules):
            options_at_module = self.module_2_options[lm.name]
            state_for_module = state_manager.prepare_state(len(options_at_module))
            module_metrics = [ # for each task
                [ # for each option
                    [0] * len(state_for_module[i][0]) # for each state
                    for _ in range(len(state_for_module[i]))
                ] 
                for i in range(len(state_for_module))
            ]
            
            for task_idx, task in enumerate(state_for_module):
                for option_idx, (option, states) in enumerate(zip(options_at_module, task)):
                    input_quality = self.module_2_input_quality[lm.name][task_idx]
                    assert len(states) == len(input_quality), "Input quality should match number of input states"
                    lm.lm_config['model'] = option
                    for state_idx, state in enumerate(states):
                        self.workflow.reset_modules()
                        self.workflow.run(
                            state=state,
                            start_from=lm,
                            stop_at=lm,
                        )
                        pred = lm.outputs[-1]
                        gold = labels[task_idx][lm.name]
                        score = self.module_2_metric[lm.name](gold, pred)
                        module_metrics[task_idx][option_idx][state_idx] = {
                            'input_quality': input_quality[state_idx],
                            'output': pred,
                            'score': score,
                            'model': option,
                        }
            module_bootstrap_profile[lm.name] = module_metrics
            new_input_quality = state_manager.update_state(state_for_module, module_metrics, self.max_sample_to_keep)
            for child in lm.children:
                self.module_2_input_quality[child.name] = new_input_quality
            
        json.dump(module_bootstrap_profile, open(profile_path, 'w+'), indent=4)
        return module_bootstrap_profile

    def bootstrap_v2(
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
                        self.workflow.reset_modules()
                        self.workflow.run(state=state, start_from=lm, stop_before=next_lm)
                        # get intermediate result
                        pred = lm.outputs[-1]
                        comparable_pred = {k: convert_to_comparable_repr(v) for k, v in pred.items()}
                        output_quality = self.module_2_metric[lm.name](gold, comparable_pred)
                        # get final output quality if this is exit point
                        if lm.name == self.workflow.exit_point[0].name:
                            final_ground_truth = {'final_output': self.trainset_label[task_idx]}
                            final_quality = self.final_output_metric(
                                final_ground_truth, 
                                {'final_output': comparable_pred[self.workflow.exit_point[1]]}
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
        def predicate(node: DecisionNode):
            return node.state_scores['final_output'] >= mean_target_final_score * gap
        solutions = [
            {'selections': p.selections, 'state_scores': p.state_scores} 
                for p in score_paths.paths if predicate(p)
        ]
        with open(solution_path, 'w+') as f:
            json.dump(solutions, f, indent=4)
        return solutions
        
    
    def compile(
        self, 
        label_path: str = 'labels.json',
        profile_path: str = 'profile.json',
        curve_path: str = 'curve.joblib',
        solution_path: str = 'solutions.json',
    ):
        """
        for all modules:
            for all options:
                for all input states:
                    generate_output
            filter_output
            output as input for next module
        """
        
        # Get ground truth labels
        labels = self.get_labels(label_path)
        
        # Get profile for each module
        module_bootstrap_profile = self.bootstrap_v2(labels, profile_path)
        
        # build curve for each option at each module from the profile
        prop_evaluator = self.fit_curve_from_profile(module_bootstrap_profile, curve_path)
        
        # Build score tree and search for the best option
        config_candidates = self.search(prop_evaluator, .95, solution_path)
        
        return config_candidates