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

Now let's walk through setting up the optimization pipeline for a single-agent system. In this example, we will define a single agent designed to answer user question using **retrieval-augmented generation (RAG)**. We will follow the three steps outlined above to perform parameter tuning for this workload.

Step 0.5: Setup the optimization environment
--------------------------------------------

To get started, let's first prepare the project file structure as follows:

1. Create a folder called ``my_project``. This will serve as the root directory for your optimization pipeline.
2. Inside the ``my_project`` folder, create the following files:
 
   - **.env**: This file will contain necessary environment variables, such as API keys.
   - **workflow.py**: This will contain the definition of your agent workflow.
   - **data_loader.py**: This file will define the dataset used to evaluate the workflow.
   - **evaluator.py**: This will contain the evaluation metric or scoring function for your workflow.
   - **control_param.py**: This file will specify the control parameters for the optimizer.

The resulting structure should look like this:

::

   my_project
   ├── .env
   ├── workflow.py
   ├── data_loader.py
   ├── evaluator.py
   └── control_param.py

In the ``.env`` file, set your openai API key as follows:

::

   OPENAI_API_KEY="YOUR_API_KEY"


Step 1: Connect to your workflow
--------------------------------

Say our original workflow is defined in *DSPy*:

.. code-block:: python

   import dspy

   colbert = dspy.ColBERTv2(url='<YOUR_URL>/api/search') # replace this with your own ColBERT server
   gpt4o_mini = dspy.LM('gpt-4o-mini', max_tokens=1000)
   dspy.configure(lm=gpt4o_mini, rm=colbert)

   class Agent(dspy.Signature):
      """
      You are an expert in responding to user questions based on provided context.
      """
      question = dspy.InputField()
      context = dspy.InputField()
      answer = dspy.OutputField()

   class QAWorkflow(dspy.Module):
      def __init__(self):
         self.retrieve = dspy.Retrieve(k=2)
         self.agent = Agent()

      def forward(self, question):
         docs = self.retrieve(question).passages
         answer = self.agent(question=question, context=docs)
         return answer

Now we want the optimizer to hook on the LLM agent in this workflow and tune its parameters. To do this, we will replace the dspy agent with `Cognify` semantics as follows:

.. code-block:: python

   from compiler.llm.model import CogLM
   from compiler.llm import InputVar, OutputLabel
   from compiler.frontends.dspy.connector import as_predict

   cognify_agent = CogLM(
      agent_name='qa_agent',
      system_prompt='You are an expert in responding to user questions based on provided context.',
      input_variables=[
         InputVar(name="question"),
         InputVar(name="context")
      ],
      output=OutputLabel(name="answer")
   )
   agent = as_predict(cognify_agent) # apply adapter for easier integration

   class QAWorkflow(dspy.Module):
      def __init__(self):
         self.retrieve = dspy.Retrieve(k=2)
         self.agent = agent

      def forward(self, question):
         docs = self.retrieve(question).passages
         answer = self.agent(question=question, context=docs)
         return answer

We will save the modified workflow in ``workflow.py``. 

.. code-block:: python

   import dspy
   from compiler.llm.model import CogLM
   from compiler.llm import InputVar, OutputLabel
   from compiler.frontends.dspy.connector import as_predict
   from compiler.optimizer import register_opt_program_entry

   colbert = dspy.ColBERTv2(url='<YOUR_URL>/api/search') # replace this with your own ColBERT server
   dspy.configure(rm=colbert)

   cognify_agent = CogLM(
      agent_name='qa_agent',
      system_prompt='You are an expert in responding to user questions based on provided context.',
      input_variables=[
         InputVar(name="question"),
         InputVar(name="context")
      ],
      output=OutputLabel(name="answer")
   )
   agent = as_predict(cognify_agent) # apply adapter for easier integration

   class QAWorkflow(dspy.Module):
      def __init__(self):
         self.retrieve = dspy.Retrieve(k=2)
         self.agent = agent

      def forward(self, question):
         docs = self.retrieve(question).passages
         answer = self.agent(question=question, context=docs)
         return answer
   
   workflow = QAWorkflow()
   
   # Also, in order to tell the optimizer how to use this workflow
   # we need to register an invoke function with the annotation
   @register_opt_program_entry
   def invoke_workflow(input):
      return workflow.forward(input)


Step 2: Build the Evaluation Pipeline
-------------------------------------

Next, we will define the data loader and evaluator for our workflow, in ``data_loader.py`` and ``evaluator.py`` respectively.

2.1 Define data loader
^^^^^^^^^^^^^^^^^^^^^^
Cognify expects a function that returns (**input / ground_truth**) pairs for the optimizer to use. These variables will be used in the following way:

::

   # workflow <- optimizer.propose()
   prediction = call_your_workflow(input)
   score = call_your_evaluator(ground_truth, prediction)
   # optimizer.update(workflow, score)

Variables like input/ground_truth/prediction will be forwarded to the corresponding functions directly without any modification. While this provides utmost flexibility in the data format, it is your responsibility to ensure function signatures match the expected input/output.

.. note::

   If your metric does not need a ground truth, e.g. using LLM judge with only scoring criteria, you are free to use any dummy value or ``None`` for the output entry. 
   
   Current optimizer will not try to inspect or exploit the ground truth information.

In this example, we provide a small subset of examples from HotPotQA dataset. The function will be registered to the optimizer as follows:

.. code-block:: python

   from compiler.optimizer.registry import register_data_loader

   @register_data_loader
   def load_data():
      data = [
         # 13 pairs of user question (input) and short answer (ground truth)
         ("""Are Walt Disney and Sacro GRA both documentry films?""", """yes"""),
         ("""What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?""", """design their own interdisciplinary program"""),
         ("""Which is published more frequently, The People's Friend or Bust?""", """The People's Friend"""),
         ("""How much is spent on the type of whiskey that 1792 Whiskey is in the United States?""", """about $2.7 billion"""),
         ("""The place where John Laub is an American criminologist and Distinguished University Professor in the Department of Criminology and Criminal Justice at was founded in what year?""", """1856"""),
         ("""What year did the mountain known in Italian as "Monte Vesuvio", erupt?""", """79 AD"""),
         ("""What was the full name of the author that memorialized Susan Bertie through her single volume of poems?""", """Emilia Lanier"""),
         ("""How many seasons did, the Guard with a FG%% around .420, play in the NBA ?""", """14 seasons"""),
         ("""Estonian Philharmonic Chamber Choir won the grammy Award for Best Choral Performance for two songs by a composer born in what year ?""", """1935"""),
         ("""Which of the sport analyst of The Experts Network is nicknamed  "The Iron Man"?""", """Calvin Edwin Ripken Jr."""),
         ("""What are both National Bird and America's Heart and Soul?""", """What are both National Bird and America's Heart and Soul?"""),
         ("""What was the 2010 population of the birthplace of Gerard Piel?""", """17,121"""),
         ("""On what streets is the hospital that cared for Molly Meldrum located?""", """the corner of Commercial and Punt Roads"""),
      ]
      train_data = data[:5]
      validation_data = None
      test_data = data[5:]
      return train_data, None, test_data

Just like dataloaders in many other frameworks (e.g. huggingface, pytorch), this function should also split the data into train/validation/test sets. In this example, we use the first 5 examples as training data, and the rest as test data. The validation set is set to ``None`` for simplicity.

2.2 Define evaluation method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cognify expects a function that takes in the ground truth and prediction, and returns a numeric score. In this example, we will use the F1 score as the evaluation metric. The function will be registered to the optimizer as follows:

.. code-block:: python

   from dsp.utils.metrics import F1

   @register_opt_score_fn
   def answer_f1(label: str, pred: str):
      if isinstance(label, str):
         label = [label]
      score = F1(pred, label)
      return score

Step 3: Configure the Optimizer Behavior
----------------------------------------

Finally, we will define the control parameters for the optimizer in ``control_param.py``. The optimizer will use these parameters to guide the search process. 

In this example, we will give a simple configuration for the optimizer. The search space has only one layer, meaning all parameters will be tuned jointly in a single optimization routine.

The parameters we want to tune for the LLM agent include 

1. the reasoning style
2. few-shot examples to add to the prompt

The optimizer will search for different combinations of these parameters to trade-off the F1 score and the cost of running the workflow.

The final configuration file will look like this:

.. code-block:: python

   from compiler.cog_hub import reasoning, fewshot
   from compiler.cog_hub.common import NoChange
   from compiler.optimizer.control_param import ControlParameter
   from compiler.optimizer.core import driver, flow

   # Define search space
   reasoning_param = reasoning.LMReasoning(
      [NoChange(), reasoning.ZeroShotCoT()] 
   )

   fewshot_param = fewshot.FewShot(max_num=4)

   # Decide parameter placement
   single_layer_config = driver.LayerConfig(
      layer_name='simple_optimization_layer',
      universal_params=[reasoning_param, fewshot_param],
   )

   # Register optimizer settings
   optimize_control_param = ControlParameter(
      opt_layer_configs=[single_layer_config],
   )


Run the Optimization
--------------------

With all the components in place, you can now run the optimization to find the most cost-efficient way to apply these prompt engineer techniques.

If you follow the naming convension in the example above, you can run the following command in the terminal:

.. code-block:: bash
   
   cd my_project
   cognify optimize workflow.py

otherwise you can specify the file names explicitly:

.. code-block:: bash

   cd my_project
   cognify optimize workflow.py -d <data_loader file name> -e <evaluator file name> -c <control_param file name>
