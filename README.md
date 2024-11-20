[Cognify-Logo](cognify.jpg)

# Cognify: A Comprehensive, Multi-Faceted Gen AI Workflow Optimizer

Building high-quality, cost-effective generative AI applications is challenging due to the absence of systematic methods for tuning, testing, and optimization. We introduce **Cognify**, a tool that automatically enhances generation quality and reduces costs for generative AI workflows, including those written with LangChain, DSPy, and annotated Python. Built on a novel foundation of hierarchical, workflow-level optimization, **Cognify** delivers up to a 56% improvement in generation quality and up to 10x cost reduction. Read more about **Cognify** [here](https://mlsys.wuklab.io/posts/cognify/).

## Installation

Cognify is available as a Python package and can be installed as
```
pip install cognify
```

Or install from the source:
```
git clone <...>
cd Cognify
pip install -e .
```

## Getting Started

You can use Cognify with our CLI:
```bash
cognify optimize workflow.py   
```
where `workflow.py` is your workflow source code. Cognify currently supports unmodified [LangChain](https://github.com/langchain-ai/langchain) and [DSPy](https://github.com/stanfordnlp/dspy) workflow source code. You can also port your existing workflow written directly on Python or develop new Python-based workflows with our [simple workflow interface]().

Additionally, Cognify automatically searches for the default three files under the same directory: `config.py`, `dataloader.py`, and `evaluator.py`. You can also specify these three files explicitly by:
```bash
cognify optimize /your/source/workflow.py -c /your/cog/config.py -d /your/sample/dataloader.py -e /your/specified/evaluator.py  
```
- **Cogs**: we define **cogs** as the various optimizations that can be applied to your workflow. These, along with hyperparameters for the optimization process, should be specified in `config.py`. Learn more about how to [configure your optimizer]().
- **Data**: the Cognify optimizer relies on training data in the form of input-output pairs. This should be specified in `dataloader.py`. Read about how to [load your data]().
- **Evaluation**: to evaluate the final workflow generation quality, you should define a scoring function in `evaluator.py`. Find out how to [evaluate your workflow]().

Read our [documentation]() to learn more.

- [Quickstart]()
- [Fundamentals]()
- [Examples]()
- [API Reference]()


## Contributing

We welcome and value any contributions to Cognify. Please read our [Contribution Instructions]() on how to get involved.
