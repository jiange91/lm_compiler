.. _cognify_interface:

The Cognify Programming Model
=============================

We introduce a simple workflow programming model, the **Cognify Programming Model**. It is an easy-to-use interface designed for implementing gen-AI workflows. You do not need to use the Cognify programming model to use Cognify. For example, you can run unmodified `LangChain <https://cognify-ai.readthedocs.io/user_guide/tutorials/interface/langchain.html>`_ and `DSPy <https://cognify-ai.readthedocs.io/user_guide/tutorials/interface/dspy.html>`_ programs on Cognify and skip this section.  

The Cognify programming model centers around :code:`cognify.Model`, a class used for defining a model call (currently, Cognify only supports language models) for Cognify's optimization.
This class is designed to be a drop-in replacement for your calls to a model such as the OpenAI API endpoints. 
:code:`cognify.Model` abstracts away the complexity of model selection and prompt construction, allowing you to focus on your business logic. 
Model calls not specified with :code:`cognify.Model` will still run but will not be optimized. 

We do not restrict how different model calls interact with each other, allowing users to freely define their relationship and communication. 
For example, you can pass the generation of one model call as the input to another model call, to multiple downstream model calls, to a tool/function calling, etc.
You can also write your own control flow like loops and conditional branching.

**Workflow Code for Solving Maths Problems:**

.. code-block:: python

   import cognify

   interpreter_prompt = """
   You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
   """
   interpreter_agent = cognify.Model("interpreter", 
      system_prompt=interpreter_prompt, 
      input_variables=[cognify.Input("problem")], 
      output=cognify.OutputLabel("math_model"),
      lm_config=cognify.LMConfig(model="gpt-4o-mini"))


   from pydantic import BaseModel
   class MathResponse(BaseModel):
      final_answer: float
      explanation: str

   solver_prompt = """
   You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
   """
   solver_agent = cognify.StructuredModel("solver",
      system_prompt=solver_prompt,
      input_variables=[cognify.Input("problem"), cognify.Input("math_model")],
      output_format=cognify.OutputFormat(MathResponse),
      lm_config=cognify.LMConfig(model="gpt-4o-mini"))

   # Define Workflow
   def math_solver_workflow(problem):
      math_model = interpreter_agent(inputs={"problem": problem})
      response: MathResponse = solver_agent(inputs={"problem": problem, "math_model": math_model})
      print(response.explanation)
      return response.final_answer


The math-solver example above has two model calls specified by :code:`cognify.Model` or :code:`cognify.StructuredModel`, the :code:`interpreter_agent` and the :code:`solver_agent`. 
The :code:`cognify.StructuredModel` class allows for more complex output formats specified outside of the class such as :code:`MathResponse`.
The :code:`math_solver_workflow` function specifies the overall workflow process, with the generation of :code:`interpreter_agent` is passed as the input of :code:`solver_agent`.

As seen, :code:`cognify.Model` encapsulates four key components that you should specify:

1. **System prompt**: The :code:`system_prompt` field specifies the initial "system" part of a prompt sequence sent to a language model to define the model's role or provide context and instructions to the model. 
For example, "You are a math problem interpreter..." and "You are a math solver..." are the system prompts for the two model calls in our math example, as shown below. 
A language model call has one system prompt that is used regardless of the user inputs. We mandate this information in the Cognify programming model because Cogs like task decomposition rely on the system prompt. 

2. **Request Inputs**: The :code:`input_variables` field specifies a dictionary of input texts from each user request. Unlike the system prompt, this field differs from workflow invocation to workflow invocation. 
Cognify allows one or more elements within :code:`input_variables`, each being one type of user input. For example, :code:`input_variables` of :code:`solver_agent` has two elements: the first being the end-user request math problem and the second being the generation of the :code:`interpreter_agent` step. 
Note that you can also concatenate the elements into one long text sequence as the only element to :code:`input_variables`.
However, the more fine-grained you can categorize your input sequences, the more likely Cognify can reach better optimization results.

3. **Output format**: The :code:`output_format` field specifies the format of the model output. 
It can simply be a label assigned to the output string or a complex schema that the response is expected to conform to. For the latter, you need to use the :code:`cognify.StructuredModel` class. 

4. **Language model configuration**: The :code:`lm_config` field specifies the initial set of language models and their configurations that Cognify uses as the starting point and as the baseline to compare for reporting its optimization improvement. You can add more models for Cognify to explore in the `optimization configuration file <https://cognify-ai.readthedocs.io/user_guide/tutorials/optimizer.html>`_. 

.. hint::

   When including models from different providers in your configuration, make sure that all required API keys are provided in your environment for the optimizer to call the models.

For Cognify to properly capture your :code:`cognify.Model`, be sure to instantiate them as global variables. 
Cognify performs parallel optimization internally for faster optimization speed.  Local instantiations of :code:`cognify.Model` will cause synchronization problems and thus will not be registered by Cognify. However, once instantiated, they can be invoked anywhere in your program.

Invoking a :code:`cognify.Model` (or :code:`cognify.StructuredModel`) is straightforward. Simply pass in a dictionary of inputs that maps the variable to its actual value. 
Cognify uses the system prompt, input variables, and output format to construct the messages to send to the model endpoints. 
We encourage users to let Cognify handle message construction and passing. However, for fine-grained control over the messages and arguments passed to the model and easy integration with your current codebase, you can optionally pass in a list of messages and your model keyword arguments. For more detailed usage instructions, check out our `GitHub repo <https://github.com/WukLab/Cognify/tree/main/cognify/llm>`_.

To integrate the workflow with Cognify, you need to register the function that invokes the workflow with our decorator ``register_opt_workflow`` like so:

.. code-block:: python

   from cognify.optimizer.registry import register_opt_workflow

   @register_opt_workflow
   def math_solver_workflow(problem):
      math_model = interpreter_agent(inputs={"problem": problem})
      answer = solver_agent(inputs={"problem": problem, "math_model": math_model})
      return answer

