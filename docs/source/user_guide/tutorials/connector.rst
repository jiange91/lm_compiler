.. _cognify_tutorials_connector:

*************
Connector
*************


The example workflow we use is a **Math Problem Solver** involving two agents called in sequence:

1. **Math Interpreter Agent**: This agent analyzes the problem and form a math model.

2. **Solver Agent**: This agent computes the solution by solving the math model generated in the previous step.

**Workflow Code:**

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   # Initialize the model
   import dotenv
   dotenv.load_dotenv()
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

   solver_prompt = """
   You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
   """

   solver_template = ChatPromptTemplate.from_messages(
      [
         ("system", solver_prompt),
         ("human", "problem:\n{problem}\n\nmath model:\n{math_model}\n"),
      ]
   )

   solver_agent = solver_template | model

   # Define Workflow
   def math_solver_workflow(problem):
      math_model = interpreter_agent.invoke({"problem": problem}).content
      answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
      return {"answer": answer}

.. note::

   We use ``gpt-4o-mini`` as the model for the original workflow. Cognify also supports tuning the model selection for each agent, which will be covered in the tutorial.

   Make sure all required API keys are provided in your environment for the optimizer to call the models.

To integrate the work with Cognify, you need to register the function that invokes the workflow with the annotation:


.. code-block:: python

   from cognify.optimizer.registry import register_opt_workflow

   @register_opt_workflow
   def math_solver_workflow(problem):
      math_model = interpreter_agent.invoke({"problem": problem}).content
      answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
      return {"answer": answer}

