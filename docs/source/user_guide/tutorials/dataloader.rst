.. _cognify_tutorials_data_loader:

*************
Data Loader
*************

The data should be formatted in ``(input / ground_truth)`` pairs.

Cognify will forward the loaded data in the following way:

.. code-block:: python

   result = registered_workflow(**input)
   score = registered_evaluator(**input, **result, **ground_truth)

.. note:: 

   The ``input``, ``result``, and ``ground_truth`` are all available to the evaluator function for convenience.
   
   The evaluator signature don't have to take in all variables, Cognify will only pass in the variables that are needed.

In this **Math Problem Solver** example, the signature of the workflow and evaluator functions are as follows:

.. code-block:: python

   def math_solver_workflow(problem):
      ...
      return {'answer': ...}

   def evaluate(problem, answer, solution):
      ...

Thus the we should format the data as follows:

::
   
   input = {
      "problem": "What is 2 + 2?",
   }
   ground_truth = {
      "solution": "4",
   }

   data_item = (input, ground_truth)
   loaded_data = [data_item, ...]

The example data loader code is as follows:

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
