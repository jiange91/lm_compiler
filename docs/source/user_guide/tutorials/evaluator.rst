.. _cognify_tutorials_evaluator:

******************
Workflow Evaluator
******************

Cognify evaluates your workflow throughout its optimization iterations. 

To tell Cognify how you want it to be evaluated, you should define an evaluator for your workflow.

.. note::
   The evaluator should **always** output a numeric value with higher being the better. You can choose your own **range** of the numeric values.

To register a function as your evaluator, simply add ``@cognify.register_evaluator`` before it.

For the math-solver example, we will use LLM-as-a-judge to be the evaluator, using Cognify's programming interface as an example:

.. code-block:: python

   import cognify

   from pydantic import BaseModel

   class Assessment(BaseModel):
      score: int
      
   evaluator_prompt = """
   You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

   You should not solve the problem by yourself, a standard solution will be provided. 

   Please rate the answer with a score between 0 and 10.
   """

   evaluator_agent = cognify.StructuredModel(
      agent_name='llm_judge',
      system_prompt=evaluator_prompt,
      input_variables=(
         cognify.Input('problem'),
         cognify.Input('solution'),
         cognify.Input('answer'),
      ),
      output_format=cognify.OutputFormat(schema=Assessment),
      lm_config=cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0}),
      opt_register=False,
   )

   @cognify.register_evaluator
   def llm_judge(problem, answer, solution):
      assess = evaluator_agent(inputs={'problem': problem, 'solution': solution, 'answer': answer})
      return assess.score

The evaluator agent uses `gpt-4o-mini` as the backbone model. It also returns a structured output, ``Assessment``, to enforce the output format.