# Cognify

**Cognify optimizes your LM-based workflow for accuracy and cost with just one click.** To complete complex tasks, LMs are often deployed as "agents" responsible for different sub-tasks within a workflow. However, tuning this workflow is not systematic and typically requires significant manual effort to tune each LM. **Enter, Cognify.**

**Cognify** hooks on to the language models used in your program and searches over various optimization techniques to give you the best configurations for both high accuracy and low cost. With **Cognify**, you can test out the latest models, prompt techniques, and even a different workflow structure. Whether you directly interact with the OpenAI API or use a framework like Langchain or DSPy, Cognify can deliver exceptional results without the need to adopt a brand new programming model and the fewest possible changes to your code.

**Cognify** outperforms the leading foundation models on a variety of tasks, able to improve accuracy by up to 56% and cost by up to 10x. See how we compare against the DSPy optimizer and leading foundation models like OpenAI o1 in our [benchmarks]().

## Installation

Cognify is available as a python package.
```
pip install cognify
```

Or, install from source.
```
git clone <...>
cd Cognify
pip install -e .
```

## Usage

You can use Cognify with our CLI like so:
```
cognify optimize /path/to/workflow.py --parameters /path/to/params.json --evaluator /path/to/evaluator.py --dataloader /path/to/dataloader.py 
```

### Parameters

The Cognify optimizer searches over a set of parameters that you can specify. There are three kinds of parameters.
1. Decomposition - breaks down a single agent's task into multiple agents
2. Reasoning - adds meta-prompting, such as Chain-of-Thought prompting and demonstrations, to assist the LM with more complex reasoning-style tasks.
3. Model Selection - chooses from different LMs, both open and closed source models

Read more about [customizing parameters]().

### Evaluators

To conduct the search, the Cognify optimizer requires an evaluator and a dataset. The dataset should be split into a training set that is used during the search and a test set that is used to evaluate the final performance. While we also provide a few evaluator functions out of the box, each workflow is different and will benefit from specialization.

Read more about [customizing evaluators]().


## Integrating Cognify

Since Cognify operates by hooking onto your LMs, very little code changes are needed. Based on your setup, there are different ways to register an LM with the Cognify optimizer.

### Completions API

If your code calls the OpenAI completions API, you can substitute the API call with our [wrapper module]() and pass all arguments just as you would directly to the endpoint. Internally, we use `litellm` to ensure consistency with the original API call. 

### Langchain or DSPy

We provide connector modules that wrap a Langchain Runnable or a DSPy Module. Optionally, you can try our [translation tool]() to automatically wrap your runnables or modules. We plan on adding support for more  frameworks in the future. If you would like to request a connector to a framework or contribute one yourself, see our [contribution guide]().

### Writing a workflow from scratch

We provide a NetworkX-style graph interface that you can use to write your workflow. You can refer to our [examples]() as a starting point. 