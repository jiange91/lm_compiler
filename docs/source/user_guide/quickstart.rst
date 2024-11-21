.. _cognify_quickstart:

******************
Cognify Quickstart
******************

This section demonstrates the basic way to use Cognify using a simple example.

Integrate Cognify in Three Steps
================================

1. Connect to Your Workflow
---------------------------

The first step of using Cognify is to connect it to your existing gen-AI workflow. 
We currently support unmodified programs written in LangChain and DSPy. 
You can also develop gen-AI workflows on our Python-based interface or modify your existing Python programs to this interface.


2. Build the Evaluation Pipeline
--------------------------------

The next step is to create an evaluation pipeline. This involves providing a training dataset and an evaluator of your workflow.

- **Training Data**: Cognify relies on user supplied training data for its optimization process. Thus, you need to provide a data loader function that returns a sequence of input/output pairs served as the ground truth. 

- **Workflow Evaluator**: We expect users (developers of workflows) to understand how to evaluate their workflows. Thus, you need to provide an evaluator function for determining the generation quality. We provide several common evaluators such as F1 and LLM-as-a-judge that you could use to start with.

3. Configure the Optimizer Behavior
-----------------------------------

The final step is to configure the optimization process. This step is optional. If not provided, Cognify will use default values to configure your optimization.
However, we highly encourage you to configure your optimization to achieve better results. You can configure your optimization in the following ways:

- **Select Model Set**: Define the set of models you want Cognify to try on your workflows. You are responsible for setting up your model API keys whenever they are needed.

- **Select Cog Search Space**: Define the *Cogs* and their *Options* you want Cognify to explore. If not set, Cognify will use a default set of Cogs.

- **Config Optimization Settings**: Establish the overall optimization strategy by defining the maximum number of search iterations, quality constraint, or cost constraint. These settings allow you to choose whether to prioritize quality improvement, cost reduction, or minimize Cognify's optimization time.

A Minimal Example
=================

Now let's walk through setting up the optimization pipeline for a naive single-agent system. 

Step 0.5: Inspect the original workflow
--------------------------------------------

To get started, let's first take a look at the original workflow that we will optimize.

.. hint::

   The code of this example is available at `examples/quickstart <https://github.com/WukLab/Cognify/tree/main/examples/quickstart>`_. This tutorial will explain each step in detail.

.. code-block:: python

   # examples/quickstart/workflow.py

   # ----------------------------------------------------------------------------
   # Define a single LLM agent to answer user question with provided documents
   # ----------------------------------------------------------------------------

   import dotenv
   from langchain_openai import ChatOpenAI
   # Load the environment variables
   dotenv.load_dotenv()
   # Initialize the model
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   # Define system prompt
   system_prompt = """
   You are an expert at answering questions based on provided documents. Your task is to provide the answer along with all supporting facts in given documents.
   """

   # Define agent routine 
   from langchain_core.prompts import ChatPromptTemplate
   agent_prompt = ChatPromptTemplate.from_messages(
      [
         ("system", system_prompt),
         ("human", "User question: {question} \n\nDocuments: {documents}"),
      ]
   )

   qa_agent = agent_prompt | model

   # Define workflow
   def doc_str(docs):
      context = []
      for i, c in enumerate(docs):
         context.append(f"[{i+1}]: {c}")
      return "\n".join(docs)

   def qa_workflow(question, documents):
      format_doc = doc_str(documents)
      answer = qa_agent.invoke({"question": question, "documents": format_doc}).content
      return {'answer': answer}

To authenticate OpenAI API, you can create a ``.env`` file in the same directory with the following content:

::

   OPENAI_API_KEY=your_openai_api_key

You can try running this workflow:

.. code-block:: python

   question = "What was the 2010 population of the birthplace of Gerard Piel?"
   documents = [
      'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
   ]

   result = qa_workflow(question=question, documents=documents)
   print(result)

**Output**:

::
   
   {'answer': 'The birthplace of Gerard Piel is Woodmere, New York. However, the provided document does not include the 2010 population of Woodmere. To find that information, one would typically refer to census data or demographic reports from that year.'}

Step 1: Connect to the workflow
-------------------------------

Cognify will automatically capture agents defined globally, we only need to inform the optimizer which function to call to run the workflow.

In this example, the entry point to the workflow is the ``qa_workflow`` function. We will register it with:

.. code-block:: python

   from cognify.optimizer import register_opt_workflow

   @register_opt_workflow
   def qa_workflow(question, documents):
      format_doc = doc_str(documents)
      answer = qa_agent.invoke({"question": question, "documents": format_doc}).content
      return {'answer': answer}

Step 2: Build the Evaluation Pipeline
-------------------------------------

Next, we will define the evaluator and training data used for optimization.

Create a ``config.py`` file under the same directory with ``workflow.py`` and define the evaluator and data loader functions there.

2.1 Register evaluator
^^^^^^^^^^^^^^^^^^^^^^

In this example, we use the ``F1`` score to quantify the similarity between the predicted answer and the ground truth.

Cognify already provides an implementation of this metric. We will register it as follows:

.. code-block:: python

   import cognify
   from cognify.optimizer.registry import register_opt_score_fn

   metric = cognify.metric.f1_score_str

   @register_opt_score_fn
   def evaluate_answer(answer, label):
      return metric(answer, label)

2.2 Register data loader
^^^^^^^^^^^^^^^^^^^^^^^^

Cognify expects the data to be formatted as (**input / ground_truth**) pairs. Both needs to be a dictionary.

In this example, we provide a small set of examples from HotPotQA dataset in :file:`data._json`. The data loader will read the file and return the pairs as follows:

.. code-block:: python

   from cognify.optimizer.registry import register_data_loader
   import json

   @register_data_loader
   def load_data_minor():
      with open("data._json", "r") as f:
         data = json.load(f)
            
      # format to (input, output) pairs
      new_data = []
      for d in data:
         input = {
               'question': d["question"], 
               'documents': d["docs"]
         }
         output = {
               'label': d["answer"],
         }
         new_data.append((input, output))
      
      # split to train, val, test
      return new_data[:5], None, new_data[5:]

.. hint::

   The dataset is small for a quick demonstration. In practice, you should provide a larger dataset for better generalization.


Step 3: Configure the Optimizer Behavior
----------------------------------------

Now we need to define a search setting for the optimizer. This should also be added to the ``config.py`` file.

The setting includes a search space and the optimization strategies. Cognify also provides a set of `pre-defined configurations <https://github.com/WukLab/Cognify/blob/main/cognify/hub/search/default.py>`_ for you to start with.

Here we just use the default one:

.. code-block:: python

   from cognify.hub.search import default

   search_settings = default.create_search()

Wrap Up
-------

Now we have all the components in place. The final directory structure should look like this:

::

   .
   ├── config.py # evaluator + data loader + search settings
   ├── data._json
   ├── workflow.py
   └── .env


Run Cognify Optimization
------------------------

.. code-block:: bash
   
   cd examples/quickstart
   cognify optimize workflow.py

**Example Output:**

.. code-block:: bash

   (my_env) user@hostname:/path/to/quickstart$ cognify optimize workflow.py 
   > light_opt_layer | (best score: 0.16, lowest cost@1000: 0.09 $): 100%|███████████████| 10/10 [01:53<00:00, 11.30s/it]
   ================ Optimization Results =================
   Num Pareto Frontier: 2
   --------------------------------------------------------
   Pareto_1
   Quality: 0.160, Cost per 1K invocation: $0.28
   Applied at: light_opt_layer_4
   --------------------------------------------------------
   Pareto_2
   Quality: 0.154, Cost per 1K invocation: $0.09
   Applied at: light_opt_layer_6
   ========================================================

The optimizer found two Pareto frontiers, meaning these two are the most cost-effective solutions within all searched ones.

.. note::

   It's not guaranteed that the optimizer will find any better solutions than the original one. You might get ``Num Pareto Frontier: 0`` in the output.

Check detailed optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find all Pareto frontiers' information under the ``opt_results/pareto_frontier_details`` directory. 

Beflow is the transformations that ``Pareto_1`` will apply to the original workflow, saying CoT reasoning is applied while no few-shot demonstration is added.

::

   Trial - light_opt_layer_4
   Log at: opt_results/light_opt_layer/opt_logs.json
   Quality: 0.160, Cost per 1K invocation ($): 0.28 $
   ********** Detailed Optimization Trace **********

   ========== Layer: light_opt_layer ==========

   >>> Module: qa_agent <<<

      - Parameter: <cognify.hub.cogs.fewshot.LMFewShot>
         Applied Option: NoChange
         Transformation Details:
         NoChange

      - Parameter: <cognify.hub.cogs.reasoning.LMReasoning>
         Applied Option: ZeroShotCoT
         Transformation Details:
         
         - ZeroShotCoT -
         Return step-by-step reasoning for the given chat prompt messages.
         
         Reasoning Prompt: 
               Let's solve this problem step by step before giving the final response.

   ==================================================


Evaluate a Specific Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can further evaluate a specific optimization on the test dataset.

.. code-block:: bash

   cognify evaluate workflow.py -s Pareto_1

**Example Output:**

.. code-block:: bash

   (my_env) user@hostname:/path/to/quickstart$ cognify evaluate workflow.py -s Pareto_1
   ----- Testing select trial light_opt_layer_4 -----
   Params: {'qa_agent_few_shot': 'NoChange', 'qa_agent_reasoning': 'ZeroShotCoT'}
   Training Quality: 0.160, Cost per 1K invocation: $0.28

   > Evaluation in light_opt_layer_4 | (avg score: 0.20, avg cost@1000: 0.28 $): 100%|███████10/10 [00:07<00:00,  1.42it/s]
   =========== Evaluation Results ===========
   Quality: 0.199, Cost per 1K invocation: $0.28
   ===========================================

You can also use Cognify to evaluate the original workflow with:

.. code-block:: bash

   cognify evaluate workflow.py -s NoChange