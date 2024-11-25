.. _cognify_dspy_interface:

DSPy
====

Cognify supports unmodified DSPy programs. All you need to do is to **register the entry function** for Cognify to execute the workflow during optimization. The following is the DSPy code for the **Math Problem Solver** example:

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
      print(response.explanation)
      return response.final_answer
    
  my_workflow = MathSolverWorkflow()

  import cognify
  
  @cognify.register_workflow
  def math_solver_workflow(problem):
    return my_workflow(problem=problem)

In DSPy, the :code:`dspy.Predict` class is the primary abstraction for obtaining a response from a language model. When intializing the optimizer, Cognify will automatically translate each predictor in your globally instantiated :code:`dspy.Module` (as defined by the module's :code:`__init__` function) into a :code:`cognify.StructuredModel`. 

.. tip::

  DSPy also contains other, more detailed modules that don't follow the behavior of :code:`dspy.Predict` (e.g., :code:`dspy.ChainOfThought`). In Cognify, we view Chain-of-Thought prompting (and other similar techniques) as possible optimizations to apply to an LLM call on the fly instead of as pre-defined templates. Hence, during the translation process we will strip the "reasoning" step out of the predictor definition and leave it to the optimizer. 
  
By default, Cognify will translate **all** predictors into valid optimization targets. For more fine-grained control over which predictors should be targeted for optimization, you can manually wrap your predictor with our :code:`cognify.PredictModel` class like so: 

.. code-block:: python

  import cognify
  import dspy

  class MathSolverWorkflow(dspy.Module):
    def __init__(self):
      super().__init__()
      self.interpreter_agent = dspy.Predict("problem -> math_model")

      # -- manually wrap the predictor with PredictModel --
      self.solver_agent = cognify.PredictModel(
        "solver_agent",
        dspy.Predict("problem, math_model -> final_answer, explanation")
      )
  
    def forward(self, problem):
      math_model = self.interpreter_agent(problem=problem).math_model
      response = self.solver_agent(problem=problem, math_model=math_model)  # unchanged
      print(response.explanation)
      return response.final_answer

  ...

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them with your existing DSPy infrastructure, you can wrap your :code:`cognify.Model` with an :code:`as_predict()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.PredictModel` and follows the :code:`dspy.Predict` protocol.

.. code-block:: python

  import cognify
  import dspy 
  
  solver_prompt = """
  You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
  """
  solver_agent = cognify.StructuredModel(
    "solver_agent",
    system_prompt=solver_prompt,
    input_variables=[cognify.Input("problem"), cognify.Input("math_model")],
    output_format=cognify.OutputFormat(MathResponse),
    lm_config=cognify.LMConfig(model="gpt-4o-mini")
  )

  class MathSolverWorkflow(dspy.Module):
    def __init__(self):
      super().__init__()
      self.interpreter_agent = dspy.Predict("problem -> math_model")
      self.solver_agent = cognify.as_predict(solver_agent)  # wrap cognify model here
  
    def forward(self, problem):
      math_model = self.interpreter_agent(problem=problem).math_model
      response = self.solver_agent(problem=problem, math_model=math_model)  # unchanged
      print(response.explanation)
      return response.final_answer

  ...

For detailed usage instructions regarding DSPy, check out our `DSPy README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/dspy>`_.