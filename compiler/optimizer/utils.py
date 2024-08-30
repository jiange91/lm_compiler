from queue import Queue
import copy
import numpy as np
from collections import defaultdict
import itertools
from typing import Tuple, Any
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

from langchain_core.documents.base import Document

from compiler.IR.program import StatePool, Module


def convert_to_comparable_repr(value):
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, Document):
        return value.page_content
    if hasattr(value, 'repr_for_quality_compare'):
        return value.repr_for_quality_compare()
    elif isinstance(value, list):
        return list(map(convert_to_comparable_repr, value))
    else:
        raise ValueError(f"Cannot convert {value} to comparable representation")


class DecisionNode:
    def __init__(self):
        self.selections = {}
        self.state_scores = {}
        

class ScorePath:
    """
    This class store the estimation of all possible config path for a program
    """
    def __init__(self, sorted_module_in_interest, module_2_option_2_predictor):
        self.sorted_module_in_interest: list[Module] = sorted_module_in_interest
        self.module_2_option_2_predictor: dict[str, dict[str, LinearPredictor]] = module_2_option_2_predictor
        self.paths = None
    
    def build_tree(self, user_input_fields: list[str]):
        initial_decision = DecisionNode()
        initial_decision.state_scores = {k: [1.0] for k in user_input_fields}
        
        frontier = Queue()
        frontier.put(initial_decision)
        for m in self.sorted_module_in_interest:
            n = len(frontier.queue)
            for _ in range(n):
                node: DecisionNode = frontier.get()
                for option in self.module_2_option_2_predictor[m.name]:
                    predictor = self.module_2_option_2_predictor[m.name][option]
                    output_field_2_score = predictor.predict(node.state_scores)
                    new_node = copy.deepcopy(node)
                    new_node.selections[m.name] = option
                    new_node.state_scores.update(output_field_2_score)
                    frontier.put(new_node)
        self.paths = list(frontier.queue)
            
    
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
    
# For each task, keep the top-k states that maximize variance
def max_var_subset(k, scores) -> list[int]:
    if len(scores) <= k:
        return [i for i in range(len(scores))]
    
    sort_ids = list(np.argsort(scores))
    max_var = float('-inf')
    choice = None
    for k0 in range(1, k):
        smaller = [scores[i] for i in sort_ids[:k0]]
        larger = [scores[i] for i in sort_ids[-(k-k0):]]
        subset = smaller + larger
        variance = np.var(subset)
        if variance > max_var:
            max_var = variance
            choice = k0
    subset_ids = sort_ids[:choice] + sort_ids[-(k-choice):]
    return subset_ids


class StateManager_v2:
    QualityType = dict[str, float]
    StateScoreType = Tuple[StatePool, QualityType]
    
    def __init__(self, trainset: list[StatePool]):
        # structure of state_by_task:
        # task: [instance1, instance2, ...]
        #   List of pairs (state, scores)
        # NOTE: state is not in comparable format as this is used as input to workflow
        self.state_score_by_task: list[list[StateManager_v2.StateScoreType]] = []
        for task in trainset:
            self.state_score_by_task.append([(task, {k: 1.0 for k in task.state.keys()})])
                
    # NOTE:
    # currently select max_sample_to_keep states that maximize variance for each field
    # e.g. max_sample_to_keep = 4, len(input_fields) = 2, then we have max 4 * 4 configs in total
    #      selected states might have overlap so not exactly 4 * 4
    # TODO: more effective variance maximze algorithm for multiple fields
    def prepare_state(
        self,
        input_fields: list[str],
        max_sample_to_keep_for_each_field: int,
    ) -> list[list[StateScoreType]]:
        pairs_by_task = []
        for state_scores in self.state_score_by_task:
            selected_pair_idx = []
            for field in input_fields:
                idx_with_field = [i for i, ss in enumerate(state_scores) if field in ss[1]]
                scores_of_field = [state_scores[i][1][field] for i in idx_with_field]
                selected_idx = max_var_subset(
                    max_sample_to_keep_for_each_field, 
                    scores_of_field
                )
                selected_pair_idx.extend(idx_with_field[i] for i in selected_idx)
            selected_pairs = [state_scores[i] for i in set(selected_pair_idx)]
            pairs_by_task.append(selected_pairs)
        return pairs_by_task

    def update_state(self, new_state_scores_by_task: list[list[StateScoreType]]):
        self.state_score_by_task = new_state_scores_by_task
        
            
class OptionProfiler:#
    """
    Note: Option profiler will copy the state in case module has inplace state update
    """
    def __init__(self, option, state_score_list):
        self.option = option
        self.input_state_score_list = state_score_list 
        self.profile_record = [] # for curve fitting
        self.new_state_score = [] # for state update
    
    def prepare_input_state_pool(self):
        for state, score in self.input_state_score_list:
            input_state = copy.deepcopy(state)
            input_score = copy.deepcopy(score)
            yield input_state, input_score
    
    # NOTE: Option runner should be callable and return (output, quality)
    def profie(self, option_runner):
        for state_pool, scores in self.prepare_input_state_pool():
            output, quality = option_runner(state_pool)
            self.profile_record.append({
                'input_quality': scores,
                'output_quality': quality,
            })
            merged_scores = scores.copy()
            merged_scores.update(quality)
            self.new_state_score.append((state_pool, merged_scores))

class LinearPredictor:
    def __init__(self):
        self.feature_columns = None
        self.predictors = {} # each output field will have a linear model
        self.average = {}
    
    def __repr__(self) -> str:
        return f'average: {self.average}, feature_columns: {self.feature_columns}, predictors: {self.predictors}'
        
    def fit(self, input_field_2_score, output_field_2_score):
        if len(input_field_2_score) == 0:
            for output_field, scores in output_field_2_score.items():
                self.average[output_field] = [np.mean(scores)]
            return
        
        self.feature_columns = list(input_field_2_score.keys())
        X = pd.DataFrame(input_field_2_score)[self.feature_columns].values
        
        for output_field, scores in output_field_2_score.items():
            y = np.array(scores)
            model = LinearRegression().fit(X, y)
            self.predictors[output_field] = model
    
    def predict(self, input_field_2_score):
        if self.feature_columns is None:
            return self.average
        
        result = {}
        for output_field, model in self.predictors.items():
            X = pd.DataFrame(input_field_2_score)[self.feature_columns].values
            result[output_field] = list(model.predict(X))
        return result
        

class PropagationEvaluator:
    def __init__(self, module_profile):
        self.module_profile = module_profile
        self.module_2_option_2_predictor = defaultdict(dict)
    
    def train(
        self, 
        modules_of_interest: list[Module],
        module_2_options: dict[str, list[str]],
    ):
        for m in modules_of_interest:
            input_fields = m.input_fields
            for option_idx, option in enumerate(module_2_options[m.name]):
                profiles = self.module_profile[m.name][option_idx]
                input_field_2_score = defaultdict(list)
                output_field_2_score = defaultdict(list)
                for profile in profiles:
                    input_quality = profile['input_quality']
                    output_quality = profile['output_quality']
                    for k, v in input_quality.items():
                        input_field_2_score[k].append(v)
                    for k, v in output_quality.items():
                        output_field_2_score[k].append(v)
                # Clean input field
                selected_fields = []
                for field, scores in input_field_2_score.items():
                    if field in input_fields:
                        if len(set(scores)) > 1:
                            selected_fields.append(field)
                predictor = LinearPredictor()
                predictor.fit(
                    {k: input_field_2_score[k] for k in selected_fields},
                    output_field_2_score
                )
                self.module_2_option_2_predictor[m.name][option] = predictor
    
    def dump(self, path):
        joblib.dump(self.module_2_option_2_predictor, path)
    
    def load(self, path):
        self.module_2_option_2_predictor = joblib.load(path)
