.. _cognify_tutorials_evaluator:

****************
Workflow Evaluator
****************

Cognify evaluates your workflow throughout its optimization iterations. To tell Cognify how you want it to be evaluated, you should define an evaluator for your workflow.
The evaluator should output a numeric value with higher being the better. You can choose your own range of the numeric values.
To register a function as your evaluator, simply add ``@cognify.register_evaluator`` before it.

For the math-solver example, we will use LLM-as-a-judge to be the evaluator, with the following LangChain implementation:

.. code-block:: python

   import cognify

   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   @cognify.register_evaluator
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
      evaluator_langchain = evaluator_template | ChatOpenAI(model="gpt-4o")
      score = evaluator_langchain.invoke({"problem": problem, "answer": answer, "solution": solution}).content
      return int(score)

