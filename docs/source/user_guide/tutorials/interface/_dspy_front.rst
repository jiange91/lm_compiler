.. code-block:: python

    import dspy

    class MathSolverWorkflow(dspy.Module):
        def __init__(self):
            super().__init__()
            self.interpreter_agent = dspy.Predict("problem -> math_model")
            self.solver_agent = dspy.Predict("problem, math_model -> final_answer, explanation")
        
        def forward(self, problem):
            math_model = self.interpreter_agent(problem=problem).math_model
            response = self.solver_agent(problem=problem, math_model=math_model)
            return response.final_answer
        
    my_workflow = MathSolverWorkflow()

    import cognify
    
    @cognify.register_workflow
    def math_solver_workflow(problem):
        return my_workflow(problem=problem)