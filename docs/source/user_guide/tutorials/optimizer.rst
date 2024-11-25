.. _config_search:

Configuring Optimizations
=========================

Cognify uses a set of configurations for its optimizations, including the maximum number of optimization iterations, the set of models to use.

Specify the Model Set
---------------------

To provide the optimizer with a list of models to search over, you can define a list of :code:`cognify.LMConfig` objects like so:

.. code-block:: python

    # config.py
    import cognify

    model_configs = [
        # OpenAI models
        cognify.LMConfig(model='gpt-4o-mini'),
        cognify.LMConfig(model='gpt-3.5-turbo-1106'),
        # Fireworks model
        cognify.LMConfig(
            custom_llm_provider='fireworks_ai',
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            cost_indicator=0.8,
            kwargs={'temperature': 0.7}
        ),
    ]

You can also set a :code:`cost_indicator` for each :code:`LMConfig` to tell the optimizer how to reason between them. By default, each :code:`LMConfig` has a :code:`cost_indicator = 1.0`, which tells the optimizer that all models are equally expensive (i.e. not to factor cost into its search). If you want Cognify to consider different models with different execution costs, you can set the :code:`cost_indicator` to different values. 

.. note::

    The :code:`cost_indicator` does not need to reflect the true difference in prices between models. For example, Llama-3.1-8b may not be 20% cheaper than GPT-4o-mini, even though we have set the cost indicator to 0.8. In this way, you can express how much you `care` about the difference in price. If you are hosting models yourself, you can set the cost according to the relative GPU resources required by each model, which usually corresponds to model weight size.

Configure Optimizer Settings
----------------------------

By default, Cognify uses a universal set of configurations for its optimization.

.. code-block:: python

    from cognify.hub.search import default
    search_settings = default.create_search(
        model_selection_cog=model_configs # pass in the model we want to search over
    )

To further customize your workflow optimization process and get the best out of Cognify, you should change a set of configurations in your :code:`create_search` function:

.. code-block:: python

    def create_search(
        *,
        opt_log_dir: str = "opt_results",
        model_selection_cog: list[LMConfig] | None = None,
        search_type: Literal["light", "medium", "heavy"] = "light",
        n_trials: int = 10,
        quality_constraint: float = 1.0,
        evaluator_batch_size: int = 10,
    )

Important parameters:

* :code:`opt_log_dir (str)`: This is the directory where the optimization results will be stored. From here, you can load the optimized workflow and use it in your code. You can also resume an optimization from the logs in this directory.
* :code:`model_selection_cog (list[LMConfig])`: Here, you can specify the models that the optimizer is allowed to search over. Ensure that you have the appropriate API key for the providers you are using. If this parameter is not specified, the optimizer will simply use the models defined by the ``LMConfig`` in the original workflow. Specifying this parameter will override the models in the original workflow.
* :code:`search_type (str)`: Either **"light", "medium",** or **"heavy"**. This determines the complexity of the optimization process, with "light" being the simplest and the one that yields the quickest results and "heavy" being the most complex and the one that yields the strongest results.
* :code:`n_trials (int)`: A trial represents one execution of a workflow on all the training data during the optimization process. This parameter allows you to roughly budget your optimization. For heavier search, we recommend a higher number of trials (e.g., 30) to allow the optimizer to effectively explore the search space.

.. note::

    As you may notice, we do not provide a default set of models to select from. This is because we do not assume any API keys are provided.

    If you want to tune the model selection with Cognify, please following the `above step <#specify-the-model-set>`_ to define the model selection cog explicitly.

Other parameters you can specify:

* :code:`quality_constraint (float)`: This represents the quality of the optimized workflow `relative to the original program`. A value of 1.0 (the default) means that the optimized workflow must be at least the same quality as the original program. If you are comfortable with slightly lower quality, you can set this value to be less than 1.0. This may allow the optimizer to find cheaper options. On the other hand, if you want a certain level of quality improvement, you can set this value to be slightly greater than 1.0. However, there is no guarantee that this solution exists. 
* :code:`evaluator_batch_size (int)`: This tells the optimizer how many training data points to execute the workflow on at once. If you are using a cloud-based service, you can adjust this parameter to avoid rate limiting.

We also provide a few built-in domain-specific configurations that you can use directly for the `example workflows <https://github.com/WukLab/Cognify/tree/main/examples>`_ we provide, including QA :code:`qa`, code generation :code:`codegen`, and data visualization :code:`datavis`. You can use these settings like:

.. code-block:: python

    from cognify.hub.search import codegen
    search_settings = codegen.create_search()
