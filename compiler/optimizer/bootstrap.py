from typing import Union, Optional
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
from compiler.optimizer.tree import TreeNode, ScoreTree

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

class StateManager:
    def __init__(self, trainset: list[StatePool]):
        self.initial_states = trainset
        self.state_by_task = [[task] for task in trainset]
    
    def prepare_state(self, num_lm_options: int):
        # duplicate states for each lm option
        dup_states = []
        for i, task in enumerate(self.state_by_task):
            dup_tasks = [copy.deepcopy(task) for _ in range(num_lm_options)]
            dup_states.append(dup_tasks)
        # the result format is:
        # [state1, state2 | state1, state2 | ...]
        # [option 1       | option 2       | ...]
        # [task 1                          | ...]
        return dup_states
    
    # NOTE: will return a list of selected score for each task as the input quality
    def update_state(self, new_states, metrics, max_sample_to_keep) -> list[list]:
        flatten_states = [[s for option in task for s in option] for task in new_states]
        flatten_metrics = [[metric['score'] for option in task for metric in option] for task in metrics]
        
        # For each task, keep the top-k states that maximize variance
        # Return the index right after smaller states
        def max_var_subset() -> list[int]:
            k = max_sample_to_keep
            if len(flatten_metrics[0]) <= k:
                return [[i for i in range(len(flatten_metrics[0]))] for _ in flatten_metrics]
            else:
                indices = []
                for task in flatten_metrics:
                    sort_ids = np.argsort(task)
                    max_var = float('-inf')
                    choice = None
                    for k0 in range(1, k):
                        smaller = [task[i] for i in sort_ids[:k0]]
                        larger = [task[i] for i in sort_ids[-(k-k0):]]
                        subset = smaller + larger
                        variance = np.var(subset)
                        if variance > max_var:
                            max_var = variance
                            choice = k0
                    indices.append(sort_ids[:choice] + sort_ids[-(k-choice):])
            return indices
        
        selected_indices = max_var_subset()
        new_states = [[flatten_states[task_id][i] for i in indices] for task_id, indices in enumerate(selected_indices)]
        self.state_by_task = new_states
        new_input_quality = [[flatten_metrics[task_id][i] for i in indices] for task_id, indices in enumerate(selected_indices)]
        return new_input_quality
    

class BootStrapLMSelection(BootStrap):
    def __init__(
        self,
        workflow: Workflow,
        teachers: Union[dict[LLMPredictor, str], str],
        module_2_options: Union[dict[LLMPredictor, list[str]], list[str]],
        module_2_metric: Union[dict[LLMPredictor, callable], callable],
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
        self.teachers = teachers
        # The maximum number of output with different qualities to keep after each module
        # This param is per traning-input
        self.max_sample_to_keep = max_sample_to_keep
        self.sorted_target_modules: list[LLMPredictor] = self.workflow.sort(lambda x: isinstance(x, LLMPredictor))
        self.module_2_input_quality = {}
        self.score_tree = None
    
    def get_labels(self, trainset: list[StatePool], label_path: str):
        if os.path.exists(label_path):
            logger.info(f"Loading labels from {label_path}")
            with open(label_path, 'r') as f:
                labels = json.load(f)
            assert len(labels) == len(trainset), "Each task should have a ground truth"
            # TODO: add more assertio to check label integrity
        else:
            # Get labels using teacher model
            for lm in self.sorted_target_modules:
                lm.lm_config['model'] = self.teachers[lm.name]
            trainset_cpy = copy.deepcopy(trainset)
            labels = [] # idx -> {lm_name, output}
            for state in trainset_cpy:
                self.workflow.reset_modules()
                self.workflow.run(state)
                labels.append({lm.name: copy.deepcopy(lm.outputs[-1]) for lm in self.sorted_target_modules})
                
            logger.info(f"Labels: {labels}")
            with open(label_path, 'w+') as f:
                json.dump(labels, f, indent=4)
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
                logger.info(f"input {task_idx}")
                for option_idx, (option, states) in enumerate(zip(options_at_module, task)):
                    input_quality = self.module_2_input_quality[lm.name][task_idx]
                    assert len(states) == len(input_quality), "Input quality should match number of input states"
                    lm.lm_config['model'] = option
                    logger.info(f"Running {lm.name} with {option}")
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
    
    def fit_curve_from_profile(self, module_bootstrap_profile, curve_dir):
        module_2_option_curve = {}
        if os.path.exists(curve_dir):
            logger.info(f"Loading curve from {curve_dir}")
            meta_json = json.load(open(f"{curve_dir}/meta.json", 'r'))
            for lm in self.sorted_target_modules:
                module_2_option_curve[lm.name] = {}
                for option in self.module_2_options[lm.name]:
                    curve = joblib.load(f"{curve_dir}/{meta_json[lm.name][option]['curve_path']}")
                    module_2_option_curve[lm.name][option] = {'curve': curve, 'r2_score': meta_json[lm.name][option]['r2_score']}
            return module_2_option_curve
        
        for lm in self.sorted_target_modules:
            module_2_option_curve[lm.name] = {}
            for option_idx, option in enumerate(self.module_2_options[lm.name]):
                # get training pairs
                input_quality, output_score = [], []
                for task in module_bootstrap_profile[lm.name]:
                    for state in task[option_idx]:
                        assert len(state['input_quality']) == 1, "assume only one input"
                        assert len(state['output']) == 1, "assume only one output"
                        input_quality.append([next(iter(state['input_quality'].values()))])
                        output_score.append(next(iter(state['score'].values())))
                # fit curve
                curve = LinearRegression().fit(input_quality, output_score)
                r2_score = curve.score(input_quality, output_score)
                module_2_option_curve[lm.name][option] = {'curve': curve, 'r2_score': r2_score}
        logger.info(f"Module 2 option curve: {module_2_option_curve}")
        
        # serialize it
        Path(curve_dir).mkdir(parents=True, exist_ok=True)
        meta_json = {}
        for lm in self.sorted_target_modules:
            meta_json[lm.name] = {}
            for option in self.module_2_options[lm.name]:
                meta_json[lm.name][option] = {
                    'curve_path': f'{lm.name}_{option}_curve.joblib',
                    'r2_score': module_2_option_curve[lm.name][option]['r2_score'],
                }
                joblib.dump(
                    module_2_option_curve[lm.name][option]['curve'],
                    f"{curve_dir}/{lm.name}_{option}_curve.joblib",
                )
        json.dump(meta_json, open(f"{curve_dir}/meta.json", 'w+'), indent=4)
        return module_2_option_curve
                        
    def search(self, module_2_option_curve, gap):
        self.score_tree = ScoreTree('user query', 1.0)
        frontier = Queue()
        frontier.put(self.score_tree.root)
        
        # build scoring tree
        for m_idx, lm in enumerate(self.sorted_target_modules):
            n = frontier.qsize()
            for _ in range(n):
                node = frontier.get()
                for option in self.module_2_options[lm.name]:
                    curve = module_2_option_curve[lm.name][option]['curve']
                    pred = curve.predict([[node.score]])[0]
                    new_node = self.score_tree.add_new_estimation(node, option, pred, 
                                                                  m_idx == len(self.sorted_target_modules) - 1)
                    frontier.put(new_node)
        solutions = self.score_tree.get_path(lambda x: x.score > 0)
        print(solutions)
        return solutions
        
    
    def compile(
        self, 
        trainset: list[StatePool],
        label_path: str = 'labels.json',
        profile_path: str = 'profile.json',
        curve_dir: str = 'curve',
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
        labels = self.get_labels(trainset, label_path)
        
        # Get profile for each module
        module_bootstrap_profile = self.bootstrap(trainset, labels, profile_path)

        # build curve for each option at each module from the profile
        module_2_option_curve = self.fit_curve_from_profile(module_bootstrap_profile, curve_dir)
        
        # Build score tree and search for the best option
        config_candidates = self.search(module_2_option_curve, 0)
        
        return config_candidates