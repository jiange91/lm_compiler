.. _cognify_interface:

DSPy
====

Cognify supports unmodified DSPy programs. All you need to is to register the entry function for Cognify to execute.

same as the langchain one, give math example, and explain what cognify transfers

In DSPy, the :code:`dspy.Predict` class is the primary abstraction for obtaining a response from a language model. A predictor contains a :code:`dspy.Signature`, from which we infer the system prompt, input variables, and output label. In DSPy, the language model is globally configured in :code:`dspy.settings`. The translation process will operate on an entire DSPy :code:`dspy.Module`, converting each :code:`dspy.Predict` into :code:`cognify.PredictModel`. Just like with Cognify models, we will only translate predictors that are instantiated in the module's `__init__` function. If you want more control over which predictors should be targeted for optimization, you can manually wrap your predictor with our :code:`cognify.PredictModel` class. DSPy also contains other, more detailed modules that don't follow the behavior of :code:`dspy.Predict` (e.g., :code:`dspy.ChainOfThought`). In Cognify, we view Chain-of-Thought prompting (and other similar techniques) as possible optimizations to apply to an LLM call on the fly instead of as pre-defined templates. Hence, during the translation process we will strip the "reasoning" step out of the predictor definition and leave it to the optimizer. For detailed usage instructions, check out our `DSPy README <https://github.com/WukLab/Cognify/tree/main/cognify/frontends/dspy>`_.

.. code-block:: python

  import dspy
  import cognify

  class MultiHopQA(dspy.Module):
    def __init__(self, passages_per_hop=3):
      super().__init__()

      self.retrieve = dspy.Retrieve(k=passages_per_hop)
      self.initial_generate_query = cognify.PredictModel(
        "initial_generate_query", 
        dspy.Predict("question -> search_query")
      )  # this is all automatically done during translation
      self.following_generate_query = cognify.PredictModel(
        "following_generate_query", 
        dspy.Predict("question, context -> search_query")
      )
      self.generate_answer = cognify.PredictModel(
        "generate_answer",
        dspy.Predict("question, context -> answer")
      )
    
    def forward(self, question):
      ### invocation code remains unchanged
      search_query = self.initial_generate_query(question=question).search_query  
      ...

If you prefer to define your modules using our :code:`cognify.Model` interface but still want to utilize them in DSPy, you can wrap your :code:`cognify.Model` with an :code:`as_predict()` call. This will convert your :code:`cognify.Model` into a :code:`cognify.PredictModel` and follows the DSPy :code:`Module` protocol. You can check out our `RAG QA tutorial <https://github.com/WukLab/Cognify/blob/main/examples/HotPotQA/tutorial.ipynb>`_ to see this in practice.

.. code-block:: python

  import cognify

  my_cog_agent = cognify.Model(
    system_prompt="You are an assistant that can summarize documents.",
    input_variables=cognify.Input("document"),
    output=cognify.OutputLabel("summary"),
    lm_config=cognify.LMConfig(model="gpt-4o-mini", max_tokens=100)
  )

  class BasicQA(dspy.Module):
    def __init__(self):
      super().__init__()
      self.generate_answer = cognify.as_predict(my_cog_agent) ### wrap cognify model here

    def forward(self, document):
      ### invocation code remains unchanged
      return self.generate_answer(document=document).answer  
