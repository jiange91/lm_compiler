.. _cognify_introduction:

***************
What is Cognify
***************

Cognify: A Comprehensive, Multi-Faceted Gen AI Workflow Optimizer
=================================================================

Building high-quality, cost-effective generative AI applications is challenging due to the lack of systematic methods for tuning, testing, and optimizing workflows. Cognify addresses this need by providing a powerful tool that automatically enhances generation quality and reduces costs in generative AI workflows, including those built with frameworks like LangChain, DSPy, and Python.

Today’s generative AI applications often rely on complex workflows rather than single model calls. These workflows can include various modules, such as generative model calls, API requests, database queries, and other code executions. They are commonly represented in graphical interfaces or programming frameworks, leading to complex configurations that are costly and difficult to optimize.

Despite the rising popularity of such gen AI workflows, manual optimization remains error-prone and resource-intensive. Existing tools typically focus on isolated modules or generation quality, overlooking the opportunity to optimize workflow architecture itself—potentially missing significant quality and cost improvements. In response, Cognify takes a holistic approach, focusing not only on individual module optimization but also on the structure and composition of entire workflows.

Key Features of Cognify
------------------------

- **Multi-Objective Optimization**: Cognify balances both quality and cost by exploring a variety of configurations, providing users with optimized workflows along the Pareto frontier of quality and cost.
- **Architecture Tuning**: Unlike traditional optimizers, Cognify can modify workflow structures by adding, removing, reordering, and parallelizing modules. This approach can significantly improve both execution cost and output quality.
- **Hierarchical Optimization Framework**: Cognify separates tuning options into architecture-maintaining (AM) and architecture-changing (AC) categories. This hierarchical model enhances efficiency, focusing on high-impact structural changes first, then fine-tuning each configuration.

Cognify currently supports workflows written in LangChain, DSPy, and annotated Python. Users simply submit their workflow code, an evaluator, and sample inputs. Cognify then compiles the workflow, producing intermediate representations optimized for different quality-cost trade-offs. These representations can be executed directly or mapped back to the original format, making Cognify a versatile tool for modern generative AI needs.

By combining comprehensive optimization techniques and multi-framework support, Cognify improves generative AI workflows' quality by up to 56% while reducing costs by as much as 10x, setting a new standard in gen AI workflow management.
