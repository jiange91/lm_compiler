![Cognify-Logo](./cognify.jpg)

# Cognify: A Comprehensive, Multi-Faceted Gen AI Workflow Optimizer

Building high-quality, cost-effective generative AI applications is challenging due to the absence of systematic methods for tuning, testing, and optimization. We introduce **Cognify**, a tool that automatically enhances generation quality and reduces costs for generative AI workflows, including those written with LangChain, DSPy, and annotated Python. Built on a novel foundation of hierarchical, workflow-level optimization, **Cognify** delivers up to a 56% improvement in generation quality and up to 10x cost reduction. Read more about **Cognify** [here](https://mlsys.wuklab.io/posts/cognify/).

## Installation

Cognify is available as a Python package and can be installed as
```
pip install cognify-ai
```

Or install from the source:
```
git clone https://github.com/WukLab/Cognify
cd Cognify
pip install -e .
```

## Getting Started

You can use Cognify with our CLI:
```bash
cognify optimize workflow.py   
```
where `workflow.py` is your workflow source code. Cognify currently supports unmodified [LangChain](https://github.com/langchain-ai/langchain) and [DSPy](https://github.com/stanfordnlp/dspy) workflow source code. You can also port your existing workflow written directly on Python or develop new Python-based workflows with our [simple workflow interface](./cognify/llm/README.md).

Cognify automatically searches for a `config.py`. You can also specify this file explicitly by:
```bash
cognify optimize workflow.py -c /your/cog/custom_config.py
```

Within the `config.py`, you should define the following:
- **Data**: the Cognify optimizer relies on training data in the form of input-output pairs. This should be specified in `dataloader.py`. <!--Read about how to [load your data]().-->
- **Evaluation**: to evaluate the final workflow generation quality, you should define a scoring function in `evaluator.py`. <!--Find out how to [evaluate your workflow]().-->
- **Search**: You can choose between light, medium, or heavy search over the optimization space. Alternatively, we provide a few application-specific search techniques for the QA, code generation, and data visualization examples provided in the repo. <!--Learn more about how to [configure search]().-->

Our optimizer searches for a fixed number of trials and then saves the best results to a checkpoint, which can then be loaded in your code for future use using `cognify.load_workflow()`. If you'd like to run more trials, you can add the `-r` or `--resume` flag like so:
```bash
cognify optimize workflow.py -r
```

For more details, check out our completeion [documentation](https://cognify-ai.readthedocs.io/en/latest/). 

<!-- Follow our [quickstart]() or read our [documentation]() to learn more.

- [Quickstart]()
- [Fundamentals]()
- [Examples]()
- [API Reference]() -->


<!-- ## Contributing

We welcome and value any contributions to Cognify. Please read our [Contribution Instructions]() on how to get involved. -->
