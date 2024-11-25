.. _cognify_tutorials_evaluator:

******************
Workflow Evaluator
******************

Cognify evaluates your workflow throughout its optimization iterations. 

To tell Cognify how you want it to be evaluated, you should define an evaluator for your workflow. 
The evaluator can be customized, but we recommend it follows the following format:

.. code-block:: python

   def evaluate(llm_prompt, llm_output, ground_truth):
      # your evaluation logic here
      return score

.. note::
   The evaluator ``score`` should **always** output a positive numerical value, where higher is better. You can choose your own **range** of the numeric values.

The evaluator function signature is customizable. For example, your evaluation function may only need the language model output and ground truth to return the score.
You can also change the order of the function parameters and their names.

To register a function as your evaluator, simply add ``@cognify.register_evaluator`` before it.

For the math-solver example, we will use LLM-as-a-judge to be the evaluator, using Cognify's programming interface as an example. 
We have modified the evaluator function signature to make the parameters more descriptive for this example:

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

The evaluator agent uses `gpt-4o-mini` as the backbone model. It also returns a structured output, ``Assessment``, to enforce the output format since we require the evaluator to return a numerical value.

.. tip::

   By default, the :code:`opt_register` value will be set to ``True`` when initializing a :code:`cognify.Model` or :code:`cognify.StructuredModel` to register the model as an optimization target in your workflow. 
   However, when calling either class in the evaluator, you should pass in :code:`opt_register=False` to avoid registering the model as an optmization target.
   