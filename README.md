# Cognify: A Comprehensive, Multi-Faceted Gen AI Workflow Optimizer

Building high-quality, cost-effective generative AI applications is challenging due to the absence of systematic methods for tuning, testing, and optimization. We introduce **Cognify**, a tool that automatically enhances generation quality and reduces costs for generative AI workflows, including those written with LangChain, DSPy, and annotated Python. Built on a novel foundation of hierarchical, workflow-level optimization, **Cognify** delivers up to a 56% improvement in generation quality and up to 10x cost reduction. Read more about **Cognify** [here]().

## Installation

Cognify is available as a Python package.
```
pip install cognify
```

Or install from the source.
```
git clone <...>
cd Cognify
pip install -e .
```

## Basic Usage

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

## CogHub

**CogHub** is a registry of gen AI workflow optimizations, what we call **cog**s. We currently support five cogs: 

* Task Decomposition: break a task into multiple more precise subtasks
* Task Ensemble: multiple workers making an ensemble of generations, from which the most consistent majority one is chosen
* Multi-Step Reasoning: asking models to reason step by step (e.g., Chain-of-Thought)
* Few-Shot Examples: adding a few high-quality example demonstrations from the sample dataset
* Model Selection: evaluating different ML models

We welcome community contributions of more cogs, which you can learn more about [here]().


## Contributing

We welcome and value any contributions to Cognify. Please read our [contribution instructions]() on how to get involved.
