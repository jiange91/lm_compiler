.. _cognify_tutorial_interface_langchain:

LangChain
=========

Cognify supports unmodified LangChain programs. All you need to do is to **register the entry function** for Cognify to execute the workflow during optimization. The following is the LangChain code for the **Math Problem Solver** example:


.. code-block:: python

  from langchain_openai import ChatOpenAI
  from langchain_core.prompts import ChatPromptTemplate

  # Initialize the model
  model = ChatOpenAI(model="gpt-4o-mini")

  interpreter_prompt = """
  You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
  """

  interpreter_template = ChatPromptTemplate.from_messages(
    [
      ("system", interpreter_prompt),
      ("human", "problem:\n{problem}\n"),
    ]
  )

  interpreter_agent = interpreter_template | model

  from langchain_core.output_parsers import PydanticOutputParser
  from pydantic import BaseModel

  class MathResponse(BaseModel):
    final_answer: float
    explanation: str
  
  parser = PydanticOutputParser(pydantic_object=MathResponse)


  solver_prompt = """
  You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
  """

  solver_template = ChatPromptTemplate.from_messages(
    [
      ("system", solver_prompt),
      ("human", "problem:\n{problem}\n\nmath model:\n{math_model}\n"),
    ]
  )

  solver_agent = solver_template | model | parser

  # Define and register workflow
  import cognify

  @cognify.register_workflow
  def math_solver_workflow(problem):
    math_model = interpreter_agent.invoke({"problem": problem}).content
    answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
    return {"answer": answer}

In LangChain, the :code:`Runnable` class is the primary abstraction for executing a task. When initializing the optimizer, Cognify will automatically translate all runnables that are instantiated as global variables into a :code:`cognify.Model` or :code:`cognify.StructuredModel`. 

.. tip::

  Your chain must follow the following format: :code:`ChatPromptTemplate | ChatModel | (optional) OutputParser`. This provides :code:`cognify.Model` with all the information it needs to optimize your workflow. The chat prompt template **must** contain a system prompt and at least one input variable. 

By default, Cognify will translate **all** runnables into valid optimization targets. For more fine-grained control over which :code:`Runnable` should be targeted, you can manually wrap your chain with our :code:`cognify.RunnableModel` class like so: 

.. code-block:: python

  import cognify
  ...

  solver_agent = solver_template | model | parser

  # -- manually wrap the chain with RunnableModel --
  solver_agent = cognify.RunnableModel("solver_agent", solver_agent)

  from cognify.optimizer.registry import register_opt_workflow
  @cognify.register_workflow
  def math_solver_workflow(problem):
    math_model = interpreter_agent.invoke({"problem": problem}).content

    # -- invocation remains the same --
    answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
    return {"answer": answer}

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them with your existing LangChain infrastructure, you can wrap your :code:`cognify.Model` with an :code:`as_runnable()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.RunnableModel` and follows the LangChain :code:`Runnable` protocol.

.. code-block:: python

  import cognify
  ...

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

  # -- manually wrap the cognify model with `as_runnable()` --
  solver_agent = cognify.as_runnable(solver_agent)

  @cognify.register_workflow
  def math_solver_workflow(problem):
    math_model = interpreter_agent.invoke({"problem": problem}).content

    # -- invocation remains the same --
    answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
    return {"answer": answer}

Cognify is also compatible with **LangGraph**, a popular orchestration framework. It can be used to coordinate LangChain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our decorator to **register** your invocation function.

For detailed usage instructions regarding LangChain and LangGraph, check out our `LangChain README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/langchain>`_.