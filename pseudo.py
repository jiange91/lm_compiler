from dataclasses import dataclass
from typing import Any

def best_score():
    ...

def best_cost(frontier, quality_constraint):
    ...

def global_trial_score_cost_monitor():
    ...

@dataclass
class EvalResult:
    scores: list[float]
    costs: list[float]
    demos: list[Demo]

class AIWorkFlowOptimization:
    def __init__(
        self, 
        evaluator,
        outer_loop,
        inner_loop,
        total_trials,
        quality_constraint,
        outer_inner_trial_ratio=1/2,
    ):
        self.evaluator = evaluator
        self.outer_optimizer = outer_loop
        self.inner_optimizer = inner_loop
        self.total_trials = total_trials
        self.quality_constraint = quality_constraint
        self.outer_inner_trial_ratio = outer_inner_trial_ratio
    
    def optimize(self, program):
        n_outer_trial = self.total_trials * self.outer_inner_trial_ratio
        n_inner_trial = self.total_trials / n_outer_trial
        
        with global_trial_score_cost_monitor() as monitor:
            for i in range(n_outer_trial):
                new_graph = self.outer_optimizer.propose(program)
                outer_loop_performance: list[tuple[float, float]] = []
                for j in range(n_inner_trial):
                    """
                    After decomposition, optimization parameters and options of each parameter
                    can be inherited optionally
                    
                    e.g. node A_params: [
                        model selection: [model 1, model 2], 
                        demonstrations: [few-shot 1, few-shot 2],
                        reasoning: [style 1, style 2]
                    ]
                    
                    decompose to A_1, A_2, inherit all parameters, but reasoning does not inherit options:
                    e.g. node A_1_params: [
                        model selection: [model 1, model 2], 
                        demonstrations: [empty],
                        reasoning: [style 1, style 2]
                    ]
                    """
                    new_graph = self.inner_optimizer.propose(new_graph)
                    eval_result: EvalResult = self.evaluator.evaluate(new_graph)
                    for param in self.inner_optimizer.dynamic_params:
                        """
                        Update dynamic parameters whose option can evolve during optimization
                        """
                        param.evolution(eval_result)
                    self.inner_optimizer.train_model(eval_result.avg_score, eval_result.avg_cost)
                    
                """
                Educate outerloop optimizer how this configuration performs
                
                Use the pareto frontier, i.e. list[(score, cost)] as indicator 
                """
                outer_loop_performance.extend(self.inner_optimizer.pareto_frontier)
                outer_score_indicator = best_score(outer_loop_performance)
                outer_cost_indicator = best_cost(outer_loop_performance, self.quality_constraint)
                self.outer_optimizer.train_model(outer_score_indicator, outer_cost_indicator)
        
            return monitor.best_trail