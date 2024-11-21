.. _cognify_introduction:

***************
What is Cognify
***************

Cognify: A Comprehensive, Multi-Faceted Gen AI Workflow Optimizer
=================================================================

Today’s generative AI (gen AI) applications often involve creating pipelines of tasks known as gen-AI workflows.
These workflows can include various components, such as gen AI model calls, tool calling, data retrieval, and other code executions. 
A key difficulty in developing gen-AI workflows is the manual testing and optimization phase, which is extremely time-consuming.
Moreover, the manually tuned workflows often have subpar generation quality or bloated execution cost.

Cognify is a fully automated gen-AI workflow optimization tool providing multi-faceted workflow optimizations.
Cognify performs workflow-level optimization by evaluating and optimizing an entire workflow instead of at each individual workflow component.
By combining comprehensive optimization techniques and multi-framework support, Cognify improves generative AI workflows' quality by up to 56% while reducing costs by as much as 10x, setting a new standard in gen AI workflow management.

To use Cognify, users supply their workflow source code, a sample input dataset, and an evaluator for determining the generation quality.
Cognify transforms the user-supplied workflow into a set of optimized versions with different quality-cost combinations that users can choose from.
These optimized versions are expressed in an intermediate representation (IR) that users can executed directly or use as the state to continue with more optimizations.


Key Features
------------

- **Multi-Objective Optimization**: Cognify provides multiple optimized workflow versions with different quality-cost combinations for users to choose from.
- **Architecture Tuning**: Unlike traditional optimizers, Cognify can modify workflow structures by adding, removing, reordering, and parallelizing modules. This approach can significantly improve both execution cost and output quality.
- **Hierarchical Optimization Framework**: Cognify separates tuning options into architecture-maintaining (AM) and architecture-changing (AC) categories. This hierarchical model enhances efficiency, focusing on high-impact structural changes first, then fine-tuning each configuration.

Benchmark Results
-----------------

We compare Cognify to non-optimized workflows and `DSPy <https://github.com/stanfordnlp/dspy>`_ using the `HotpotQA <https://hotpotqa.github.io/>`_ workload,
a `code generation <https://github.com/openai/human-eval>`_ workload, and a `text-to-SQL <https://github.com/ShayanTalaei/CHESS>`_ workload.
The figures below show the generation quality and execution cost effectiveness (larger the better for both) of these results.
Cognify pushes the cost-quality Pareto frontier over DSPy and non-optimized workflows across these workloads,
achieving 3.7% to 27% quality improvements and 1.8x to 7x cost reduction.

.. image:: /_static/images/hotpotqa.png
    :alt: HotpotQA workload
    :width: 30%

.. image:: /_static/images/codegen.png
    :alt: Code Generation workload
    :width: 30%

.. image:: /_static/images/chess.png
    :alt: Text-2-sql workload
    :width: 30%