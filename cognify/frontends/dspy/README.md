# DSPy

In DSPy, the `dspy.Predict` class is the primary abstraction for obtaining a response from a language model. A predictor contains a `dspy.Signature`, from which we infer the system prompt, input variables, and output label. In DSPy, the language model is globally configured in `dspy.settings`. The translation process will operate on an entire `dspy.Module` (i.e., a workflow), converting each `dspy.Predict`s into a `cognify.PredictModel`. As with DSPy, we will only translate predictors that are instantiated in the module's `__init__()`; users should refrain from initializing new predictors in the module's `forward()`.

For more control over which predictors are optimized, pass the `--no-translate` flag to the `$ cognify optimize` command. Then, manually connect a DSPy `dspy.Predict` to Cognify by wrapping your `dspy.Predict` with our wrapper class `cognify.PredictModel`:
```python
import dspy
import cognify

# ... setup dspy lm and retriver ...

class SingleHop(dspy.Module):
  def __init__(self):
    self.retrieve = dspy.Retrieve(k=3)
    self.generate_answer = cognify.PredictModel(
      "generate_answer",
      dspy.Predict("context,question->answer"),
    ) ## wrap with cognify
  
  def forward(self, question):
    ## invocation code remains unchanged
    context = self.retrieve(question).passages
    answer = self.generate_answer(context=context, question=question)
    return dspy.Prediction(context=context, answer=answer)
```

DSPy is a tool that automatically generates prompts on behalf of the user, which we access directly at the message passing layer. A core difference between Cognify and DSPy is we treat reasoning as a cog at the optimizer level, while DSPy requires users to specify reasoning in the interface. Because the Cognify optimizer can modify the user's prompts by applying various reasoning techniques, we strip explicit reasoning from the predictor.
```python
# reasoning will be stripped from this predictor
generate_answer = cognify.PredictModel("generate_answer", dspy.ChainOfThought(BasicQA)) 
```

By default, DSPy provides structured output back to the user in the form of a `dspy.Prediction`. We preserve this behavior so your module's `forward()` can remain unchanged. Under the hood, all `cognify.PredictModel`s act as `cognify.StructuredModel`s. 