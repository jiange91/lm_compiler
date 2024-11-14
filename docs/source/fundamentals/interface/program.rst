.. _cognify_interface:

########################################
Programming with Cognify
########################################

:code:`cognify.Model`
=====================

There are 4 key components that the Cognify optimizer utilizes when improving a workflow:

1. **System prompt** - this is often used to define the model's role or provide context and instructions to the model. Cogs like task decomposition rely on the system prompt. 
2. **Input variables** - these are the parts of the message to the model that differ from user to user. For example, in a task like "summarize a document", the document to be summarized would be an input variable. Few-shot reasoning requires robust labeled examples, which is only possible if the differences between user inputs can be captured.
3. **Output format** - this can be simply a label that is assigned to the output string or a complex schema that the response is expected to conform to. Cognify uses this information across various optimizations, such as constructing few-shot examples and to maintain consistency in the output format during task decomposition.
4. **Language model config** - this tells Cognify which model is being used and what arguments should be passed to the model. The model selection cog can change the model that gets queried for a particular task.

We package all of these into a single :code:`cognify.Model` class. This class is designed to be a drop-in replacement for your calls to the OpenAI endpoint. It abstracts away the complexity of constructing messages to send to the model and allows you to focus on the task at hand. The optimizer will use the information in the :code:`cognify.Model` to construct messages to send to the model and to optimize the workflow.

.. tip::

  Many other frameworks will refer to these 4 components as an "agent". We simply call it a :code:`cognify.Model` because we see it as a wrapper around a language model that can be optimized with our cogs. 


To ensure the optimizer captures your each :code:`cognify.Model`, be sure to instantiate them as global variables. The optimizer requires a stable set of targets, so any ephemeral/local instantiations of :code:`cognify.Model` will not be registered. Once instantiated, however, they can be invoked anywhere in your program.

.. code-block:: python

  import cognify

  # this will be registered with the optimizer
  my_cog_agent = cognify.Model(
    system_prompt="You are an assistant that can summarize documents.",
    input_variables=cognify.Input("document"),
    output=cognify.OutputLabel("summary"),
    lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
  )

  def invoke_cog_agent(document: str):
    return my_cog_agent({"document": document})

  # this will NOT be registered with the optimizer
  # Cognify needs to know all models that can be used ahead-of-time
  def create_and_invoke_my_cog_agent(document: str):
    ...
    my_cog_agent = cognify.Model(
      system_prompt="You are an assistant that can summarize documents.",
      input_variables=cognify.Input("document"),
      output=cognify.OutputLabel("summary"),
      lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
    )
    return my_cog_agent({"document": document})

Invoking a :code:`cognify.Model` is straightforward. Simply pass in a dictionary of inputs that maps the variable to its actual value. The optimizer will then use the system prompt, input variables, and output format to construct the messages to send to the model endpoint. Under the hood, it calls the :code:`litellm` completions API. We encourage users to let Cognify handle message construction and passing. However, for fine-grained control over the messages and arguments passed to the model and easy integration with your current codebase, you can optionally pass in a list of messages and your model keyword arguments. For more detailed usage instructions, check out our `GitHub repo <https://github.com/WukLab/Cognify/tree/main/cognify/llm>`_.

To set up your workflow for optimization, simply decorate the workflow entry point (i.e. the function in which the workflow is invoked) with the :code:`@cognify.workflow_entry` decorator. This will notify the optimizer to invoke that function with input samples during the optimization process. For each training example, this function will run the workflow and return the final output.

The :code:`cognify.Model` is designed to replace your calls to the OpenAI endpoint. However, many users may already have written their workflow in a framework like LangChain or DSPy.

Other frameworks
================

By default, if your current program is based on either LangChain or DSPy, Cognify will automatically translate your LangChain Runnables and DSPy Predictors into :code:`cognify.Model` during the initialization step. Below, we'll go through the nuances of each framework.

LangChain
---------

In LangChain, the :code:`Runnable` class is the primary abstraction for executing a task. To create a :code:`cognify.Model` from a runnable chain, the chain must contain a chat prompt template, a chat model, and optionally an ouptut parser. The chat prompt template is used to construct the system prompt and obtain the input variables, the chat model is used to obtain the language model config, and the output parser is used to construct the output format. If no output parser is provided, Cognify will assign a default label. Just like with :code:`cognify.Model`, we will only translate runnables that are instantiated as global variables. The translation process automatically converts each runnable in the global scope into a :code:`cognify.Model`. However, if you want more control over which :code:`Runnable` should be targeted for optimization, you can manually wrap your chain with our :code:`cognify.RunnableModel` class and pass the :code:`--no-translate` flag to the :code:`$ cognify optimize` command. For detailed usage instructions, check out our `LangChain README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/langchain>`_.

.. tip::

  Your chain must follow the following format: :code:`ChatPromptTemplate | ChatModel | (optional) OutputParser`. This provides :code:`cognify.Model` with all the information it needs to optimize your workflow. The chat prompt template must contain a system prompt and at least one input variable.

.. code-block:: python

  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.chat_models import ChatOpenAI
  from langchain_core.output_parsers import StrOutputParser

  # typical langchain code
  my_prompt_template = ChatPromptTemplate([("system", "You are an assistant that can summarize documents."), ("human", "{document}")])
  my_chat_model = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
  my_output_parser = StrOutputParser()
  my_langchain = my_prompt_template | my_chat_model | my_output_parser

  ### all it takes! this is what happens under the hood during the automatic translation
  import cognify
  my_langchain = cognify.RunnableModel(my_langchain)    # you can pass `--no-translate` to manually choose which runnables to target

  def invoke_chain(document: str):
    return my_langchain.invoke({"document": document})   # invocation code remains unchanged

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them with your existing LangChain infrastructure, you can wrap your :code:`cognify.Model` with an :code:`as_runnable()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.RunnableModel` and follows the LangChain :code:`Runnable` protocol.

.. code-block:: python

  import cognify
  from cognify.frontends.langchain import as_runnable
  from langchain_core.runnables import RunnableLambda
  from langchain_core.output_parsers import StrOutputParser

  my_runnable_cog_agent = as_runnable(cognify.Model(
    system_prompt="You are an assistant that can summarize documents.",
    input_variables=cognify.Input("document"),
    output=cognify.OutputLabel("summary"),
    lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
  ))

  def invoke_chain(document: str):
    ### fits right into your existing LangChain code
    my_chain = my_runnable_cog_agent | StrOutputParser() | RunnableLambda(lambda x: len(x))
    return my_chain.invoke({"document": document})

**LangGraph** is an orchestrator that is agnostic to the underlying framework. It can be used to orchestrate LangChain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our decorator wherever you are invoking your compiled graph.

DSPy
------

In DSPy, the :code:`dspy.Predict` class is the primary abstraction for obtaining a response from a language model. A predictor contains a :code:`dspy.Signature`, from which we infer the system prompt, input variables, and output label. In DSPy, the language model is globally configured in :code:`dspy.settings`. The translation process will operate on an entire DSPy :code:`dspy.Module`, converting each :code:`dspy.Predict` into :code:`cognify.PredictModel`. Just like with Cognify models, we will only translate predictors that are instantiated in the module's `__init__` function. If you want more control over which predictors should be targeted for optimization, you can manually wrap your predictor with our :code:`cognify.PredictModel` class and pass the :code:`--no-translate` flag to the :code:`$ cognify optimize` command. DSPy also contains other, more detailed modules that don't follow the behavior of :code:`dspy.Predict` (e.g., :code:`dspy.ChainOfThought`). In Cognify, we view Chain-of-Thought prompting (and other similar techniques) as possible optimizations to apply to an LLM call on the fly instead of as pre-defined templates. Hence, during the translation process we will strip the "reasoning" step out of the predictor definition and leave it to the optimizer. For detailed usage instructions, check out our `DSPy README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/dspy>`_.

.. code-block:: python

  import dspy
  import cognify

  class MultiHopQA(dspy.Module):
    def __init__(self, passages_per_hop=3):
      super().__init__()

      self.retrieve = dspy.Retrieve(k=passages_per_hop)
      self.initial_generate_query = cognify.PredictModel(dspy.Predict("question -> search_query"))   # this is all automatically done during translation
      self.following_generate_query = cognify.PredictModel(dspy.Predict("question, context -> search_query")) # you can pass `--no-translate` to manually choose which runnables to target
      self.generate_answer = cognify.PredictModel(dspy.Predict("question, context -> answer"))
    
    def forward(self, question):
      ### invocation code remains unchanged
      search_query = self.initial_generate_query(question=question).search_query  
      ...

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them in DSPy, you can wrap your :code:`cognify.Model` with an :code:`as_predict()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.PredictModel` and follows the DSPy :code:`Module` protocol. You can check out our `RAG QA tutorial <https://github.com/WukLab/Cognify/blob/main/examples/HotPotQA/tutorial.ipynb>`_ to see this in practice.

.. code-block:: python

  import cognify
  from cognify.frontends.dspy import as_predict

  my_cog_agent = cognify.Model(
    system_prompt="You are an assistant that can summarize documents.",
    input_variables=cognify.Input("document"),
    output=cognify.OutputLabel("summary"),
    lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
  )

  class BasicQA(dspy.Module):
    def __init__(self):
      super().__init__()
      self.generate_answer = as_predict(my_cog_agent) ### wrap cognify model here

    def forward(self, document):
      ### invocation code remains unchanged
      return self.generate_answer(document=document).answer  