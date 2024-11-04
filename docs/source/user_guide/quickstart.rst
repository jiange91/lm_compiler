.. _cognify_quickstart:

******************
Cognify Quickstart
******************

Building a high-quality, cost-effective generative AI application requires a systematic approach to defining, evaluating, and optimizing workflows. Cognify enables this by providing tools to construct multi-agent workflows, evaluate them against specific metrics, and systematically tune them for both quality and cost.

Integrate Cognify into your AI app developement *Iteration* in three steps:
=============================

1. Connect to Your Workflow

The first step in using Cognify is to connect it to your existing generative AI workflow. This involves integrating Cognify’s internal representation (IR) with the original components of your workflow, allowing the optimizer to control each agent’s behavior. The flexibility of Cognify’s IR means it can adapt to various AI modules, including LangChain, DSPy, and low-level Python code.


2. Build the Evaluation Pipeline
3. 
Once the initial workflow is connected, the next step is to create an evaluation pipeline. This involves defining the data sources and performance metrics that will be used to assess the workflow’s effectiveness. The evaluation pipeline serves as a benchmark, providing the necessary data inputs and metrics to measure the quality and cost-effectiveness of the workflow’s output.

   - **Data Source**: The evaluation pipeline begins by selecting a dataset appropriate for the task, such as a question-answering dataset for QA workflows. Cognify allows users to specify a data loader function, which manages the dataset and ensures that the workflow is tested across a representative set of inputs.
   
   - **Metric Definition**: Next, a scoring function is defined to measure the accuracy or relevance of the workflow’s output. For example, in a QA workflow, this could involve calculating the F1 score between the predicted and actual answers. The evaluation metric helps quantify workflow performance, allowing the optimizer to target improvements in quality, cost, or both.



Once the initial workflow is set up, the next step is to create an evaluation pipeline. This involves defining the data sources and performance metrics that will be used to assess the workflow’s effectiveness. The evaluation pipeline serves as a benchmark, providing the necessary data inputs and metrics to measure the quality and cost-effectiveness of the workflow’s output.

   - **Data Source**: The evaluation pipeline begins by selecting a dataset appropriate for the task, such as a question-answering dataset for QA workflows. Cognify allows users to specify a data loader function, which manages the dataset and ensures that the workflow is tested across a representative set of inputs.
   
   - **Metric Definition**: Next, a scoring function is defined to measure the accuracy or relevance of the workflow’s output. For example, in a QA workflow, this could involve calculating the F1 score between the predicted and actual answers. The evaluation metric helps quantify workflow performance, allowing the optimizer to target improvements in quality, cost, or both.

3. Configure the Optimizer Behavior
-----------------------------------

With the workflow and evaluation pipeline in place, the final step is configuring the optimizer to refine the workflow’s performance. Cognify’s optimizer is highly flexible, enabling both fine-grained parameter tuning and larger structural adjustments to the workflow.

   - **Parameter Optimization**: The optimizer begins by tuning key parameters within each agent, such as the prompt format, few-shot examples, or reasoning style. By iterating through different configurations, the optimizer aims to maximize the workflow’s quality metrics while balancing cost.

   - **Structural Optimization**: Cognify’s optimizer also supports architecture-level changes, such as modifying the composition of the workflow itself. This includes adding, removing, or reordering agents to explore cost-effective configurations without sacrificing performance. By tuning the structure, Cognify addresses both component-specific and workflow-level efficiencies, maximizing output quality and cost savings.

   - **Multi-Objective Optimization Configuration**: The optimizer allows users to specify objectives like generation quality, execution cost, or a blend of both. These configurations shape how the optimizer explores possible configurations, setting the stage for multi-iteration tuning that incrementally improves workflow performance.

---

This high-level overview describes the complete process of setting up, evaluating, and optimizing a workflow with Cognify. By following these steps, users can systematically develop robust generative AI applications that deliver high-quality results while remaining cost-efficient.