.. _cognify_tutorials_evaluator:

****************
Workflow Evaluator
****************

Cognify accept a python function to be registered as the evaluator with ``@register_opt_score_fn``.

.. note::

   The function should return a numeric value. There's no range requirement for the score, but it should be consistent across different evaluations.

You can even use the LLM as a judge in the evaluation function. In this example we will ask a LLM agent to grade the answer.

.. code-block:: python

   from cognify.optimizer.registry import register_opt_score_fn

   @register_opt_score_fn
   def evaluate(problem, answer, solution):
      evaluator_prompt = """
      You are a math problem evaluator. Your task is to grade the answer to a math problem by assessing its correctness and completeness.

      You should not solve the problem by yourself; a standard solution will be provided. 

      Please only respond with the score number, which should be a number between 0 and 10. No additional text is needed.
      """
      evaluator_template = ChatPromptTemplate.from_messages(
         [
            ("system", evaluator_prompt),
            ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n"),
         ]
      )
      evaluator_agent = evaluator_template | model
      score = evaluator_agent.invoke({"problem": problem, "answer": answer, "solution": solution}).content
      return int(score)

