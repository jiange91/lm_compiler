.. _cognify_cog:

###
Cog
###

Introduction
============

In Cognify, workflows are composed of modular components, and `Cogs` define learnable parameters within these modules. Each `Cog` has a set of options that can be applied to change the module behavior. The optimizer selects the best option for each `Cog` based on the evaluation results.

.. tip::

   Think of each Cog in Cognify as similar to a learnable weight in a neural network. Just as neuron weight influence how a neural network predicts, Cogs control various aspects of a generative AI workflow by setting different value for each Cog, allowing it to adapt for optimal performance.

Key Concepts
============

- **Cog**: A parameter that needs to be optimized within a module.
- **Option**: A choice or setting available to a `Cog`, each offering different transformations to the module.
- **Dynamic Cog**: A `Cog` capable of evolving during optimization, allowing more advanced adaptability in the options it carries based on evaluation results.

Cog: The Unit of Optimization
-----------------------------

A `Cog` always contains the following attributes:

- **name** (`str`): The name of the `Cog`, developers can use this to identify the `Cog` easily. **NO** two `Cogs` should share the same name if they are in the same module.
- **options** (`list[OptionBase]`): A list of options available for this `Cog`, where each option represents a different way of transforming the module.
- **default_option** (`Union[int, str]`): The default option for the `Cog`, specified by index or name.
- **module_name** (`str`): The name of the module where the `Cog` is applied.
- **inherit** (`bool`): Specifies whether current `Cog` can be inherited by new modules derived from the current one during the optimization process. Set to `True` by default.

Option: Module Behavior Descriptor
----------------------------------

Each `Option` encapsulates a unique configuration or transformation for a module. Selecting an option within a `Cog` changes the module’s behavior according to the option’s implementation. The following core information is included in each `Option`:

- **name** (`str`): The name of the option. This helps developers select or reference specific configurations easily. **NO** two options should share the same name within a `Cog`.
- **cost_indicator** (`float`): A pre-evaluation estimate of the relative execution cost of applying this option.  
   - **Purpose**: Helping the optimizer anticipate the expense of evaluating a configuration. This is especially useful when two options are expected to have a similar effect on quality, allowing the optimizer to favor a lower-cost option for a more efficient search.
   - **Scope**: The `cost_indicator` only provides a rough estimation to guide frugal search decisions, but it doesn’t replace actual execution costs. You may even set it to a large value to discourage using an option, while the optimizer still relies on real execution costs for final assessment.
   - **Usage**: You can override :func:`OptionBase._get_cost_indicator` to customize the cost penalty for each option. By default it returns `1.0`.

   .. rubric:: Example

   If a module originally costs `$0.5` to execute, and applying this option is expected to increase it to `$1.5`, a reasonable `cost_indicator` would be `3`.

In addition to these attributes, each `Option` provides an ``apply`` method that performs the actual transformation on the module. This method is responsible for changing the module based on the option’s configuration.

DynamicCog: Adaptive Parameter in Cognify
-----------------------------------------

A `DynamicCog` is a specialized type of `Cog` in Cognify that can evolve or adapt its options based on evaluation results during the optimization process. Unlike standard `Cogs`, which have a fixed set of options, `DynamicCogs` are designed to update or generate new options dynamically in response to performance feedback. 

Apart from standard attributes in normal `Cog`, each `DynamicCog` includes an `evolve` method, which defines how the `Cog` should adapt based on evaluation results. This method is customizable, allowing developers to tailor the evolution process to suit specific parameter types or optimization goals.

**Benefits of Using DynamicCog**

The adaptability of `DynamicCogs` allows for more granular control over parameters that benefit from dynamic refinement. By enabling parameters to evolve based on evaluation feedback, `DynamicCogs` make Cognify’s optimization process more efficient and effective, particularly for complex workflows requiring iterative improvements.

**Note**: The exact behavior of a `DynamicCog` depends on how the developer implements the `evolve` method. This customization provides flexibility, allowing `DynamicCogs` to be tailored to various types of parameters.

.. rubric:: Example

A practical usecase of a `DynamicCog` is the `few-shot` parameter (available in Cognify's cog hub), which uses high-scoring examples (demonstrations) to improve language model behavior. Here’s how it works:

1. **Initialization**: The `few-shot` `DynamicCog` begins with a set of initial options, often based on demonstrations from dry-run or an empty set.

2. **Tracking by Data Points**: The `few-shot` `DynamicCog` monitors the highest score achieved by each data point in the evaluation dataset. Each data point provides a unique demonstration of how to solve the task, and the data points with the highest scores are most likely to be effective examples.

3. **Updating Top-K Examples**: When a data point achieves a new high score and is within top-K (K is pre-defined), the `few-shot` `DynamicCog` generates a new option with the current top-K highest-scoring data points as demonstrations to guide the model’s behavior.

Out-of-the-box Cogs in Cognify
========================================

Cognify provides a set of pre-built `Cogs` that can be added to the search space or be applied directly to your workflows. Current available `Cogs` in Cognify are all LLM centric, designed to optimize the performance of agents in a generative AI workflow. Their target module is ``CogLM``.

Model Selection
---------------

The **Model Selection Cog** (`LMSelection`) in Cognify enables the adjustment of language models for each agent within a workflow. This Cog allows the optimizer to choose between different model configurations to balance quality and cost based on the task's requirements. Each model configuration is encapsulated within a `ModelOption`.

ModelOption
~~~~~~~~~~~

Each `ModelOption` defines a unique language model configuration with the following key properties:

- **model_config** (`LMConfig`): Contains the configuration details for the model, such as the provider (`openai`, `fireworks`, etc.), model name, built-in cost indicator, and other standard parameters (e.g., `max_tokens`, `temperature`).
- **cost_indicator** (`float`): A property that reads the cost indicator from :attr:`LMConfig.cost_indicator`, helping the optimizer evaluate cost-effectiveness.
- **apply** (`LLMPredictor`): A method that applies the model configuration to an `LLMPredictor` module, updating it with the selected model settings and reinitializing the predictor if necessary.

Example Usage
~~~~~~~~~~~~~

Below is an example of how to define and initialize a Model Selection Cog with multiple model options:

.. code-block:: python

   from compiler.cog_hub.model_selection import LMSelection, model_option_factory
   from compiler.IR.llm import LMConfig

   # Define model configurations, each encapsulated in a ModelOption
   model_configs = [
      # OpenAI model
      LMConfig(
         provider='openai',
         model='gpt-4o-mini',
         cost_indicator=1.0,
         kwargs={'temperature': 0.0}
      ),
      # Fireworks model
      LMConfig(
         provider='fireworks',
         model="accounts/fireworks/models/llama-v3p1-8b-instruct",
         cost_indicator=0.6,
         kwargs={'temperature': 0.0}
      ),
      # Self-hosted model with OpenAI-compatible API
      LMConfig(
         provider='local',
         model='llama-3.1-8b',
         cost_indicator=0.0,  # Indicates no cost for local models
         kwargs={
            'temperature': 0.0,
            'openai_api_base': 'http://192.168.1.16:30000/v1'
         }
      ),
   ]

   # Create Model Options from LM configurations
   options = model_option_factory(model_configs)

   # Initialize the Model Selection Cog; the optimizer will search from the above options
   model_selection_cog = LMSelection(
      name="model_selection_example",
      options=options,
   )


LLM Agent Reasoning Cog
-----------------------

The **Reasoning Cog** (`LMReasoning`) in Cognify introduces reasoning steps to the LLM generation, allowing agents to produce responses conditioned on rationale tokens. This Cog provides varies methods to enhance the quality and interpretability of responses, especially in complex tasks that require clear logic and multi-step problem-solving.


ReasonThenFormat Methodology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Cognify, all reasoning options follow the **ReasonThenFormat** approach, designed to leverage the full potential of reasoning tokens without compromising the quality of the model's output. Traditional implementations often generate reasoning tokens alongside the main output, typically in a structured JSON format where one key contains the reasoning tokens and another contains the original response. However, this approach has several limitations.

1. Generating reasoning tokens and final output in a single pass can diminish the model’s generation capability. Existing research suggests that constraining generation with strict formatting requirements can degrade output quality, particularly when models are required to follow a specific structure, such as JSON.
2. Embedding reasoning tokens within a structured format can complicate format instructions, especially if the original output itself has certain formatting requirements, such as Pydantic models or other structured responses.

.. rubric:: Cognify's approach

To address these limitations, Cognify adopts the **ReasonThenFormat** methodology. This approach separates reasoning generation from final output generation, allowing models to produce reasoning tokens freely before synthesizing a structured response.

1. **Free Generation of Reasoning Tokens**: In the first LLM call, the model generates reasoning tokens without any formatting constraints, preserving the model's generative capacity and encouraging more detailed and coherent reasoning.

2. **Concatenation and Final Output**: In the second LLM call, the reasoning tokens are appended to the original prompt, along with any specific output formatting instructions required for the final response. This lets the model synthesize a formal answer based on both the initial prompt and the freely generated reasoning tokens, ensuring that the final output is both well-reasoned and formatted as needed.

.. rubric:: Implementation

Cognify’s Intermediate Representation (IR) allows flexible control over output instructions. During the reasoning step, we remove any formatting constraints (e.g., “be concise,” “output in JSON format”) to avoid interference with reasoning quality. Once the reasoning tokens are generated, we append them to the original prompt and apply the output instructions only in the final call.

.. note::
   This method requires two consecutive LLM calls—one for reasoning tokens and one for the formatted output. However, prompt tokens from the reasoning call are often cacheable (a feature supported by many providers, including OpenAI and Anthropic), which mitigates the cost and overhead of the additional call.


ZeroShotCoT
~~~~~~~~~~~

The `ZeroShotCoT` option implements `Zero-Shot Chain-of-Thought <https://arxiv.org/pdf/2205.11916>`_, guiding the model to reason through a problem step-by-step before providing a final answer. This approach is useful for tasks that require multi-step reasoning or vertical problem-solving.

- **Cost Indicator**: By default 2.0. the extra reasoning step incurs moderate cost.
- **Reasoning Instruction**: "Let's solve this problem step by step before giving the final response."
  
PlanBefore
~~~~~~~~~~

The `PlanBefore` option encourages the model to break down a task into sub-tasks, providing responses for each sub-task as part of the reasoning process. This process largely resembles the agent architecture proposed in `LLMCompiler <https://arxiv.org/pdf/2205.11916>`_, which is originally designed to accelerate task execution. This approach is useful for complex questions that can be decomposed into smaller, parallel queries.

- **Cost Indicator**: By default 3.0. This is a modest estimation (assuming 2-subtask plan in average). You can adjust it based on the complexity of the task.
- **Reasoning Instruction**: "Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them."

Other Reasoning Options
~~~~~~~~~~~~~~~~~~~~~~~

In addition to **ZeroShotCoT** and **PlanBefore**, Cognify offers other options. While we won’t go into detail for each here, these options allow for further customization of reasoning strategies within workflows, and more options are planned for future releases.

The other reasoning options currently available include:

- **Tree-of-Thought**: Structures reasoning in a tree-like format to explore multiple solution paths. See the paper: `Tree of Thoughts: Deliberate Problem Solving with Large Language Models <https://arxiv.org/abs/2305.10601>`_.
- **Meta-Prompting**: Guides the main agent to decompose complex tasks into subtasks handled by specialized "experts", whose outputs are then coordinated and integrated by the main worker. The persona and prompt for each expert is generated by the main agent. See the paper: `Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding <https://arxiv.org/abs/2305.10601>`_.

Example Usage
~~~~~~~~~~~~~

Here is an example of how to define and initialize an LMReasoning Cog with multiple options:

.. code-block:: python

   from compiler.cog_hub.common import NoChange
   from compiler.cog_hub.reasoning import LMReasoning, ZeroShotCoT, PlanBefore

   # NoChange option stands for NO transformation to the module
   reasoning_options = [NoChange(), ZeroShotCoT(), PlanBefore()]

   # Initialize the LMReasoning Cog
   reasoning_cog = LMReasoning(
      name="reasoning_example",
      options=reasoning_options,
   )

Cognify strives to provide a comprehensive set of reasoning options to cater to various reasoning requirements in generative AI workflows. Apart from registering the reasoning Cog in the search space, you can also apply it manually to your workflow to enhance the reasoning capability of your LLM agents. 

.. code-block:: python

   from compiler.llm.model import StructuredCogLM 
   from compiler.llm import InputVar, OutputFormat
   from compiler.frontends.dspy.connector import as_predict
   from pydantic import BaseModel

   # Define the response format schema
   class Response(BaseModel):
      supporting_facts: list[str]
      answer: str

   # Initialize a StructuredCogLM
   # Cognify will automatically inject format instructions to the prompt
   cognify_agent = StructuredCogLM(
      agent_name='qa_agent',
      system_prompt='You are an expert in responding to user questions based on provided context. Answer the question and also provide supporting facts from the context.',
      input_variables=[
         InputVar(name="question"),
         InputVar(name="context")
      ],
      output_format=OutputFormat(schema=Response),
   )

   output: Response = cognify_agent.forward(
      {
         "question": "What is the capital of France?",
         "context": "France is a country in Europe."
      }
   )

   # Applying ZeroShotCoT reasoning manually to the agent
   from compiler.cog_hub.reasoning import ZeroShotCoT

   cognify_agent = ZeroShotCoT().apply(cognify_agent)
   output: Response = cognify_agent.forward(
      {
         "question": "What is the capital of France?",
         "context": "France is a country in Europe."
      }
   )

   # Inspect the reasoning step result
   print(cognify_agent.rationale)

