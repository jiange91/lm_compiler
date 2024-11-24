.. _cognify_tutorials_data_loader:

*************
Data Loader
*************

The Cognify optimization process utilizes a user-provided training dataset to `evaluate [https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/evaluator.html]`_ the workflow in iterations.
The training dataset should provide a set of inputs and the ground-truth generation outputs.
The format of the input and output should follow your workflow's needs (e.g., text and text for QA workflows, text and SQL for text-to-SQL workflows), and the exact format should match your evaluator function signature.
In each optimization iteration, Cognify runs all the data points in the training dataset to find the overall quality/cost of the optimized workflow.

.. hint::

   For more consistent, generalizable optimization results, your training dataset should be diverse enough to cover key use cases. Meanwhile, the larger your dataset is, the longer and more costly Cognify's optimization process will be. Ideally, you should provide one data point per usage category. For cases where this is hard to know, we recommend you to first try a small dataset with a few iterations and `resume <https://cognify-ai.readthedocs.io/en/latest/user_guide/tutorials/cli.html>`_ with more data and iterations.

In our math-solver example, the signature of the workflow and evaluator functions are as follows:

.. code-block:: python

   def math_solver_workflow(problem):
      ...
      return {'answer': ...}

   def evaluate(problem, answer, solution):
      ...

Thus, the training dataset should be formatted as follows:

::
   
   input = {
      "problem": "What is 2 + 2?",
   }
   ground_truth = {
      "solution": "4",
   }

   data_item = (input, ground_truth)
   loaded_data = [data_item, ...]

and the example data-loader function is as follows:

.. code-block:: python

   from cognify.optimizer.registry import register_data_loader
   import json

   @register_data_loader
   def load_data():
      with open("data._json", "r") as f:
         data = json.load(f)
            
      # format to (input, output) pairs
      new_data = []
      for d in data:
         input = {
               'problem': d["problem"],
         }
         ground_truth = {
               'solution': d["solution"],
         }
         new_data.append((input, ground_truth))
      return new_data[:5], None, new_data[:]


Cognify will forward the loaded data in the following way:

.. code-block:: python

   result = registered_workflow(**input)
   eval_inputs = as_per_func_signature(registered_evaluator, input, result, ground_truth)
   score = registered_evaluator(**eval_inputs)

.. note:: 

   The ``input``, ``result``, and ``ground_truth`` are all made available to the evaluator function for convenience.
   
   The evaluator signature don't have to consume all variables, Cognify will only pass in the variables that are needed.


