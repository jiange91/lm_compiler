.. _cognify_interface:

The Cognify Programming Model
=============================

We introduce a simple workflow programming model, the **Cognify Programming Model**. It is an easy-to-use interface designed for implementing gen-AI workflows. You do not need to use the Cognify programming model to use Cognify. For example, you can run unmodified LangChain and DSPy programs on Cognify.  

The Cognify programming model encapsulates four key components:

1. **System prompt** - The system prompt is the initial part of a structured this is often used to define the model's role or provide context and instructions to the model. Cogs like task decomposition rely on the system prompt. 
2. **Input variables** - these are the parts of the message to the model that differ from user to user. For example, in a task like "summarize a document", the document to be summarized would be an input variable. Few-shot reasoning requires robust labeled examples, which is only possible if the differences between user inputs can be captured.
3. **Output format** - this can be simply a label that is assigned to the output string or a complex schema that the response is expected to conform to. Cognify uses this information across various optimizations, such as constructing few-shot examples and to maintain consistency in the output format during task decomposition.
4. **Language model config** - this tells Cognify which model is being used and what arguments should be passed to the model. The model selection cog can change the model that gets queried for a particular task.

This class is designed to be a drop-in replacement for your calls to the OpenAI endpoint. It abstracts away the complexity of constructing messages to send to the model and allows you to focus on the task at hand. The optimizer will use the information in the :code:`cognify.Model` to construct messages to send to the model and to optimize the workflow.

.. tip::

  Many other frameworks will refer to these 4 components as an "agent". We simply call it a :code:`cognify.Model` because we see it as a wrapper around a language model that can be optimized with our cogs. 


To ensure the optimizer captures your each :code:`cognify.Model`, be sure to instantiate them as global variables. The optimizer requires a stable set of targets, so any ephemeral/local instantiations of :code:`cognify.Model` will not be registered. Once instantiated, however, they can be invoked anywhere in your program.

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

:vo integrate the work with Cognify, you need to register the function that invokes the workflow with the annotation:


.. code-block:: python

   from cognify.optimizer.registry import register_opt_workflow

   @register_opt_workflow
   def math_solver_workflow(problem):
      math_model = interpreter_agent.invoke({"problem": problem}).content
      answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
      return {"answer": answer}

Invoking a :code:`cognify.Model` is straightforward. Simply pass in a dictionary of inputs that maps the variable to its actual value. The optimizer will then use the system prompt, input variables, and output format to construct the messages to send to the model endpoint. Under the hood, it calls the :code:`litellm` completions API. We encourage users to let Cognify handle message construction and passing. However, for fine-grained control over the messages and arguments passed to the model and easy integration with your current codebase, you can optionally pass in a list of messages and your model keyword arguments. For more detailed usage instructions, check out our `GitHub repo <https://github.com/WukLab/Cognify/tree/main/cognify/llm>`_.

To set up your workflow for optimization, simply decorate the workflow entry point (i.e. the function in which the workflow is invoked) with the :code:`@cognify.workflow_entry` decorator. This will notify the optimizer to invoke that function with input samples during the optimization process. For each training example, this function will run the workflow and return the final output.

The :code:`cognify.Model` is designed to replace your calls to the OpenAI endpoint. However, many users may already have written their workflow in a framework like LangChain or DSPy.
