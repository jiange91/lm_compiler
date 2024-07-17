from typing import Union
import copy

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.modules import LMConfig, LLMPredictor

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
    
    def prepare_state_for_input(self, num_lm_options: int):
        # duplicate states for each lm option
        for task in self.state_by_task:
            states_in_task = len(task)
            for i in range(states_in_task):
                task.extend(copy.deepcopy(task[i]) for _ in range(num_lm_options - 1))
    
    def add_states(self, updates: list):
        pass
    

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
            teachers = {m: teachers for m in workflow.modules}
        if not isinstance(module_2_options, dict):
            module_2_options = {m: module_2_options for m in workflow.modules}
        if not isinstance(module_2_metric, dict):
            module_2_metric = {m: module_2_metric for m in workflow.modules}
            
        self.workflow = workflow
        self.module_2_options = module_2_options
        self.module_2_metric = module_2_metric
        self.teachers = teachers
        # The maximum number of output with different qualities to keep after each module
        # This param is per traning-input
        self.max_sample_to_keep = max_sample_to_keep
        
        self.sorted_target_modules: list[LLMPredictor] = self.workflow.sort(lambda x: isinstance(x, LLMPredictor))
    
        
    def bootstrap(self, trainset: list[StatePool]):
        """
        for all modules:
            for all options:
                for all input states:
                    generate_output
            filter_output
            output as input for next module
        """
        
        # Get labels using teacher model
        for lm in self.sorted_target_modules:
            lm.lm_config['model'] = self.teachers[lm]
        
        labels = [] # idx -> {lm, output}
        for state in trainset:
            self.workflow.reset_modules()
            self.workflow.run(state)
            labels.append({lm: copy.deepcopy(lm.outputs[-1]) for lm in self.sorted_target_modules})
        print(labels)
        
        
        module_options_curve = []
        
        for lm in self.sorted_target_modules:
            for option in self.module_2_options[lm]:
                self.workflow.reset_modules()
                lm.lm_config['model'] = option
                for state in trainset:
                    pass