.. _cognify_quickstart:

******************
Cognify Quickstart
******************

Building a high-quality, cost-effective generative AI application requires a systematic approach to defining, evaluating, and optimizing workflows. Cognify enables this by providing tools to construct multi-agent workflows, evaluate them against specific metrics, and systematically tune them for both quality and cost.

Integrate Cognify in **3** Steps
================================

1. Connect to Your Workflow
---------------------------

The first step in using Cognify is to connect it to your existing generative AI workflow. This involves integrating Cognify’s internal representation (IR) with the original components of your workflow, allowing the optimizer to control each agent’s behavior. The flexibility of Cognify’s IR means it can adapt to various AI modules, including LangChain, DSPy, and low-level Python code.


2. Build the Evaluation Pipeline
--------------------------------

Once the initial workflow is connected, the next step is to create an evaluation pipeline. This involves defining the data sources and performance metrics that will be used to assess the workflow.

- **Data Source**: The evaluation pipeline begins by selecting a dataset appropriate for the task, such as a question-answering dataset for QA workflows. Cognify allows users to register a data loader function that returns a sequence of input/output paris for evaluation.

- **Metric Definition**: A scoring function is defined to measure the accuracy or relevance of the workflow’s output. The evaluation metric helps quantify workflow performance, allowing the optimizer to reason about how to search different configurations.

3. Configure the Optimizer Behavior
-----------------------------------

With the workflow and evaluation pipeline in place, the final step is configuring the optimizer to refine the workflow’s performance. Cognify’s optimizer is highly flexible, enabling both fine-grained parameter tuning and larger structural adjustments to the workflow.

- **Select Search Space**: Define the *Cogs* and their *Options* that the optimizer will explore, also known as the search space.

- **Decide Cog Placement**: Determine at which optimization *layer* each *Cog* should be placed. This decision impacts the update frequency at each Cog dimension and the stability of the optimization process.

- **Config Optimization Settings**: Establish the overall optimization strategy by defining the settings for search iterations, resource allocation policy and quality constraints. Normally users can control the trade-off between optimization cost and result by adjusting the configuration here.

A Minimal Example
=================

Now let's walk through setting up the optimization pipeline for a single-agent system. In this example, we will define a single LLM agent to answer user question using the provided context.

.. hint::

   The complete code and data for trying this example can be found at `examples/quickstart <https://github.com/WukLab/Cognify/tree/add_doc_cog/examples/quickstart>`_. This tutorial will explain each step in detail.

Step 0.5: Setup the optimization environment
--------------------------------------------

To get started, let's first inspect the project file structure:

1. The folder ``quickstart`` serves as the root directory for your optimization pipeline.
2. Inside the folder, the following files are included:
 
   - **.env**: This file will contain necessary environment variables, such as API keys.
   - **ori_workflow.py**: The primitive AI workflow defined in pure LangGraph.
   - **cognify_workflow.py**: The workflow to be optimized. Modified with Cognify semantics for the optimizer to hook on.
   - **data_loader.py**: This file will load the dataset used to evaluate the workflow.
   - **evaluator.py**: This will contain the evaluation metric or scoring function for your workflow.
   - **control_param.py**: This file will specify the control parameters for the optimizer.
   - **data._json**: A small set of data points. This file is **optional**, include here for the clarity of ``data_loader.py``.

.. note::

   ``.env`` file is ommitted in the repo, please create the file and set your API keys.

   ``ori_workflow.py`` is not required and is not the workflow to be optimized. Include this file to show how to connect an existing workflow to Cognify.

The overall structure looks like this:

::

   quickstart/
   ├── .env
   ├── cognify_workflow.py
   ├── control_param.py
   ├── data_loader.py
   ├── data._json
   ├── evaluator.py
   └── ori_workflow.py


Step 1: Connect to the workflow
-------------------------------

1.1 Original workflow
^^^^^^^^^^^^^^^^^^^^^
Our initial workflow is defined in :file:`ori_workflow.py`. Let's take a closer look at the definition of the llm agent:

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from pydantic import BaseModel
   from typing import List


   # Define system prompt
   system_prompt = """
   You are an expert at answering questions based on provided documents. 
   Your task is to provide the answer along with all supporting facts in given documents.
   """

   # Define Pydantic model for structured output
   class AnswerOutput(BaseModel):
      answer: str
      supporting_facts: List[str]
      
   # Initialize the model
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerOutput)

   # Define agent routine 
   from langchain_core.prompts import ChatPromptTemplate
   agent_prompt = ChatPromptTemplate.from_messages(
      [
         ("system", system_prompt),
         ("human", "User question: {question} \n\nDocuments: {documents}"),
      ]
   )

   qa_agent = agent_prompt | model

The agent is backed by GPT-4o-mini. It takes in a user question and a series of documents, then returns the answer along with supporting facts. The output is structured as a Pydantic model.

You can try running this agent with:

.. code-block:: python

   print(qa_agent.invoke(
      {
         "question": "What was the 2010 population of the birthplace of Gerard Piel?", 
         "documents": """
            [1]: Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).
            [2]: Woodmere, New York | Woodmere is a hamlet and census-designated place (CDP) in Nassau County, New York, United States. The population was 17,121 at the 2010 census.
         """,
      }
   ))

**Output**:

::

   answer='The 2010 population of Woodmere, New York, the birthplace of Gerard Piel, was 17,121.'
   supporting_facts=[
      'Gerard Piel was born on 1 March 1915 in Woodmere, N.Y.', 
      'Woodmere is a hamlet and census-designated place (CDP) in Nassau County, New York.', 
      'The population of Woodmere was 17,121 at the 2010 census.'
   ]

You can further refer to :file:`ori_workflow.py` for the complete implementation.

1.2 Use Cognify semantics
^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we show how to modify this agent to connect it to the optimizer. Cognify provides rich features to define a LLM agent in a more structured way.

.. code-block:: python

   from compiler.llm.model import StructuredCogLM, InputVar, OutputFormat
   cognify_qa_agent = StructuredCogLM(
      agent_name="qa_agent",
      system_prompt=system_prompt,
      input_variables=[InputVar(name="question"), InputVar(name="documents")],
      output_format=OutputFormat(schema=AnswerOutput),
      lm_config=lm_config
   )

   # Use builtin connector for smooth integration
   from compiler.frontends.langchain.connector import as_runnable
   qa_agent = as_runnable(cognify_qa_agent)

To facilitate smooth integration with various frontend, we encourage using provided adapters (e.g. ``as_runnable``) to convert the CogLM agent interface. 

To this point, we successfully create a CogLM agent that the optimizer can transform while seamlessly fitting into the original workflow.

.. note::
   
   Auxiliary messages such as "User question: {question} \n\nDocuments: {documents}" or output format instructions will be automatically added by the Cognify runtime. This simplify the agent definition for users and grant more flexibility for the optimizer to adjust the agent behavior.

You can try running this agent with the same input.

**Output**:

::

   answer='The population of Woodmere, New York in 2010 was 17,121.' 
   supporting_facts=[
      'Gerard Piel was born in Woodmere, New York.', 
      'Woodmere is a hamlet and census-designated place (CDP) in Nassau County, New York, United States.', 
      'The population of Woodmere was 17,121 at the 2010 census.'
   ]


Step 2: Build the Evaluation Pipeline
-------------------------------------

Next, we will define the data loader and evaluator for our workflow, in ``data_loader.py`` and ``evaluator.py`` respectively.

2.1 Define data loader
^^^^^^^^^^^^^^^^^^^^^^
Cognify expects a function that returns (**input / ground_truth**) pairs for the optimizer to use. 

The ``input`` will be forwarded to the workflow directly. The the ``ground_truth`` along with the ``output`` will be forwarded to the evaluator intactly.

In short:
::

   # [(input, ground_truth), ...] <- data_loader()
   # workflow <- optimizer.propose()
   # for each pair:
      prediction = call_your_workflow(input)
      score = call_your_evaluator(ground_truth, prediction)
   # optimizer.update(workflow, score)

While this provides utmost flexibility in the data format, it is your responsibility to ensure function signatures match the expected data type.

.. hint::

   If your metric does not need a ground truth, e.g. using LLM judge with only scoring criteria, you are free to use any dummy value or ``None`` for the ground_truth. 
   
   Current optimizer will not try to inspect or exploit the ground truth information.

In this example, we provide a small set of examples from HotPotQA dataset in :file:`data._json`. The data loade  function will read the file and return the pairs as follows:

.. code-block:: python

   from compiler.optimizer.registry import register_data_loader
   import json

   @register_data_loader
   def load_data_minor():
      with open("data._json", "r") as f:
         data = json.load(f)

      # format to (input, output) pairs
      new_data = []
      for d in data:
         input = (d["question"], d["docs"])
         output = d["answer"]
         new_data.append((input, output))
      return new_data[:5], None, new_data[5:]


Just like dataloaders in many other frameworks (e.g. huggingface, pytorch), this function also split the data into train/validation/test sets. In this example, we use the first 5 rows as training data, and the rest as test data. The validation set is set to ``None`` for simplicity.

2.2 Define evaluation method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cognify expects a function that takes in the ground truth and prediction, and returns a numeric score. 

In this example, we will use the F1 score as the evaluation metric. Please check the ``evaluator.py`` file for the complete code if needed.

The function will be registered to the optimizer as follows:

.. code-block:: python

   from compiler.optimizer import register_opt_score_fn

   @register_opt_score_fn
   def f1(label: str, pred: str) -> float:
      score = f1_score_strings(label, pred)
      return score

Step 3: Configure the Optimizer Behavior
----------------------------------------

Finally, we will define control parameters for the optimizer in ``control_param.py``, including the search space and optimization settings.

In this example, we will construct a simple 2-layer search space for the optimizer to explore.

.. rubric:: Bottom-layer

The bottom-layer includes the following parameters:

1. **reasoning style**: whether to use zero-shot CoT or not
2. **few-shot examples**: teach the agent with a few good demonstrations

.. code-block:: python

   from compiler.cog_hub import reasoning, fewshot
   from compiler.cog_hub.common import NoChange

   # ================= Inner Loop Config =================
   # Reasoning Parameter
   reasoning_param = reasoning.LMReasoning(
      [NoChange(), reasoning.ZeroShotCoT()] 
   )
   # Few Shot Parameter
   few_shot_params = fewshot.LMFewShot(2)

Then we define how the optimizer should search through these parameters:

.. code-block:: python

   from compiler.optimizer.core import driver, flow

   inner_opt_config = flow.OptConfig(
      n_trials=4,
   )
   inner_loop_config = driver.LayerConfig(
      layer_name='inner_loop',
      universal_params=[few_shot_params, reasoning_param],
      opt_config=inner_opt_config,
   )

We register the search space and allow the optimizer to try 4-iterations to find the best combination at the bottom layer.

.. rubric:: Top-layer

Similarly, we define the top-layer search space and the optimizer settings as follows:

.. code-block:: python

   from compiler.cog_hub import ensemble

   # Ensemble Parameter
   general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
   general_ensemble_params = ensemble.ModuleEnsemble(
      [NoChange(), general_usc_ensemble]
   )
   # Layer Config
   outer_opt_config = flow.OptConfig(
      n_trials=2,
   )
   outer_loop_config = driver.LayerConfig(
      layer_name='outer_loop',
      universal_params=[general_ensemble_params],
      opt_config=outer_opt_config,
   )

At this layer, we will determine if `ensembling <https://arxiv.org/abs/2311.17311>`_ should be applied to the agent in two trials. If applied, multiple agents will be spawned and the final output will be a combination of their outputs.

.. note::

   Each spawned agent will be optimized independently in the bottom layer with the same search space.

   Each top-layer trial will run a complete bottom-layer optimization process, meaning the total number of workflow evaluations will be **2*4 = 8**.

.. rubric:: Overall Optimizer Settings

Finally, we define the control parameters for the optimizer, registering the 2-layer search space and decide the directory to store the optimization results:

.. code-block:: python

   from compiler.optimizer.control_param import ControlParameter

   optimize_control_param = ControlParameter(
      opt_layer_configs=[outer_loop_config, inner_loop_config],
      opt_history_log_dir='quick_opt_results'
   )

You can refer to the complete code in ``control_param.py`` for an overview. 

The optimizer will search for different combinations of these parameters to trade-off the F1 score and the cost of running the workflow.

Run the Optimization
--------------------

With all the components in place, you can now run the optimization to find the most cost-efficient way to apply these prompt engineer techniques.

If you follow the naming convension in the example above, you can run the following command in the terminal:

.. code-block:: bash
   
   cd examples/quickstart
   cognify optimize cognify_workflow.py

otherwise you can specify the file names explicitly:

.. code-block:: bash

   cd examples/quickstart
   cognify optimize cognify_workflow.py -d <data_loader file name> -e <evaluator file name> -c <control_param file name>
