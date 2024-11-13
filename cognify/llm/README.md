# Cognify Interface

When writing a workflow with Cognify, you can define your optimization targets using our `CogLM` class. Defining a `CogLM` requires four components:
1. System prompt
2. Input variables
3. Output label
4. Language model config 

The Cognify optimizer treats the system prompt as the agent's role, necessary for cogs like task decomposition. The input variables and output label are used to construct high-quality few-shot examples. When utilizing the model selection cog, the optimizer can modify the model configuration and arguments of a `CogLM`. You can also use our `StructuredCogLM` class and provide a Pydantic-based output _schema_ in lieu of an output label. `CogLM` and `StructuredCogLM` both make calls to `litellm` under the hood, so you can always expect [consistent output](https://docs.litellm.ai/docs/completion/output). Both classes also support image input. 

Much like other frameworks, we endorse the separation of the LLM's signature from its invocation. The Cognify optimizer registers your `CogLM`s at initialization, which means they should be defined in the global namespace. Otherwise, the optimizer does not have a stable set of targets from one trial to the next. You can read more about our optimizer [here]().

## Usage

Integrating Cognify into your code is straightforward.
1. Define the `CogLM` as a global variable.
2. Use our decorator to mark the workflow entry point `@cognify.workflow_entry`
3. Call your `CogLM` directly with the relevant inputs.

```python
import cognify
from cognify import InputVar, CogLM, OutputLabel, LMConfig

# define cognify agent
qa_question = InputVar(name="question")
cog_agent = CogLM(agent_name="qa_agent",
  system_prompt="You are a helpful AI assistant that answers questions.",
  input_variables=[qa_question],
  output_label=OutputLabel(name="answer"),
  lm_config=LMConfig(
    model="gpt-4o-mini", 
    kwargs={"temperature": 0.0, "max_tokens": 100}
  )
)

@cognify.workflow_entry
def call_qa_llm(question):
  return cog_agent(inputs={qa_question: question})
```

By default, `CogLM` will construct messages on your behalf based on the `system_prompt`, `inputs` and `output_label`. These messages are directly passed to model defined in the `lm_config`. For compatibility with existing codebases that rely on passing messages and keyword arguments directly, we allow the user to pass in optional `messages` and `model_kwargs` arguments when calling a `CogLM` like so:


```python
import cognify
from cognify import InputVar, CogLM, OutputLabel, LMConfig

system_prompt = "You are a helpful AI assistant that answers questions."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

# define cognify agent
qa_question = InputVar(name="question")
cog_agent = CogLM(agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question],
  output_label=OutputLabel(name="answer")
)

@cognify.workflow_entry
def call_qa_llm(question):
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": f"Answer the following question: {question}"
    }
  ]
  return cog_agent(
    inputs={qa_question: question}, 
    messages=messages, 
    model_kwargs=model_kwargs
  )
```

## Output Formatting

Cognify allows for additional output formatting. In the base `CogLM` class, you can specify custom formatting instructions when defining the output label like so:
```python
cog_agent = CogLM(
  ...
  output_label=OutputLabel(
    name="answer", 
    custom_output_format_instructions="Answer the question in less than 10 words."
  )
  ...
)
```

### Structured Output

When working with `StructuredCogLM`, you must provide a schema that will be used to format the response.
```python
from cognify import ..., StructuredCogLM, OutputFormat
from pydantic import BaseModel

class ConfidentAnswer(BaseModel):
  answer: str
  confidence: float

struct_cog_agent = StructuredCogLM(
  ...
  output_format=OutputFormat(
    schema=ConfidentAnswer
  )
  ...
)

conf_answer: ConfidentAnswer = struct_cog_agent(...)
```

The `OutputFormat` class also supports custom formatting instructions, as well as an optional hint parameter: if `should_hint_format_in_prompt=True`, Cognify will construct more detailed hints for the model based on the provided schema.

## Image Inputs

The `InputVar` class supports an optional `image_type` parameter, which can take on the values of "web", "jpeg", or "png". If either "jpeg" or "png" is selected, the system expects the image to be Base64 image upload for consistency with the [OpenAI Vision docs](https://platform.openai.com/docs/guides/vision). An image input variable can be specified like so:
```python
from cognify import InputVar
...
# Typical function to encode the image into base64
import base64
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

...
image_input = InputVar(name="my_image", image_type="png")
base64_str = encode_image("my_image_path.png")
...

response = cog_agent(inputs={..., 
                            image_input: base64_str, 
                            ...})
```