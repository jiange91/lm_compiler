.. _cognify_interface:

LangChain
=========

In LangChain, the :code:`Runnable` class is the primary abstraction for executing a task. To create a :code:`cognify.Model` from a runnable chain, the chain must contain a chat prompt template, a chat model, and optionally an ouptut parser. The chat prompt template is used to construct the system prompt and obtain the input variables, the chat model is used to obtain the language model config, and the output parser is used to construct the output format. If no output parser is provided, Cognify will assign a default label. Just like with :code:`cognify.Model`, we will only translate runnables that are instantiated as global variables. The translation process automatically converts each runnable in the global scope into a :code:`cognify.Model`. However, if you want more control over which :code:`Runnable` should be targeted for optimization, you can manually wrap your chain with our :code:`cognify.RunnableModel` class. For detailed usage instructions, check out our `LangChain README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/langchain>`_.

.. tip::

  Your chain must follow the following format: :code:`ChatPromptTemplate | ChatModel | (optional) OutputParser`. This provides :code:`cognify.Model` with all the information it needs to optimize your workflow. The chat prompt template must contain a system prompt and at least one input variable.

.. code-block:: python

  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain_core.output_parsers import StrOutputParser

  # typical langchain code
  my_prompt_template = ChatPromptTemplate([
    ("system", "You are an assistant that can summarize documents."), 
    ("human", "{document}")
  ])
  my_chat_model = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
  my_output_parser = StrOutputParser()
  my_langchain = my_prompt_template | my_chat_model | my_output_parser

  ### all it takes! this is what happens under the hood during the automatic translation
  import cognify
  my_langchain = cognify.RunnableModel("my_chain", my_langchain)

  def invoke_chain(document: str):
    return my_langchain.invoke({"document": document}).content   # invocation code remains unchanged

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them with your existing LangChain infrastructure, you can wrap your :code:`cognify.Model` with an :code:`as_runnable()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.RunnableModel` and follows the LangChain :code:`Runnable` protocol.

.. code-block:: python

  import cognify
  from langchain_core.runnables import RunnableLambda
  from langchain_core.output_parsers import StrOutputParser

  my_runnable_cog_agent = cognify.as_runnable(
    cognify.Model(
      system_prompt="You are an assistant that can summarize documents.",
      input_variables=cognify.Input("document"),
      output=cognify.OutputLabel("summary"),
      lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
    )
  )

  def invoke_chain(document: str):
    ### fits right into your existing LangChain code
    my_chain = my_runnable_cog_agent | StrOutputParser() | RunnableLambda(lambda x: len(x))
    return my_chain.invoke({"document": document}).content

**LangGraph** is an orchestrator that is agnostic to the underlying framework. It can be used to orchestrate LangChain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our decorator wherever you are invoking your compiled graph.
