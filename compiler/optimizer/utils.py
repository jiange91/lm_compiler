from queue import Queue
import copy
import numpy as np

from compiler.IR.program import StatePool

def convert_to_comparable_repr(value):
    if isinstance(value, (int, float, str, bool)):
        return value
    if not hasattr(value, 'repr_for_quality_compare'):
        raise ValueError(f"Cannot convert {value} to comparable representation")
    else:
        return value.repr_for_quality_compare()


class TreeNode:
    def __init__(self, parent, config, score):
        self.parent = parent
        self.config = config
        self.score = score
        self.next_steps = []
    
    def __repr__(self) -> str:
        return f'{{config: {self.config}, score: {self.score}}}'
        

class ScoreTree:
    """
    This class store the estimation of all possible config path for a program
    """
    def __init__(self, root_config, root_score):
        self.root = TreeNode(None, root_config, root_score)
        self.exixt_nodes = []
        
    def add_new_estimation(
        self, 
        parent: TreeNode, 
        config, 
        score,
        is_exist: bool
    ):
        new_node = TreeNode(parent, config, score)
        parent.next_steps.append(new_node)
        if is_exist:
            self.exixt_nodes.append(new_node)
        return new_node

            
    def get_path(self, predicate: callable):
        """
        Get all paths that have score within the gap
        """
        paths = []
        for node in self.exixt_nodes:
            if predicate(node):
                path = []
                while node is not None:
                    path.append(node)
                    node = node.parent
                paths.append(path[::-1])
        return paths
                
    
    
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
    
    