# Cognify Interface

The most common usage of LLMs is via direct use of the OpenAI chat completions API, like in the following example:
```python
import openai

system_prompt = "You are a helpful AI assistant built to answer questions."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

def call_qa_llm(question):
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": f"{question}"
    }
  ]

  return openai.chat.completions.create({
    messages: messages,
    model_kwargs: model_kwargs
  })
```

Much like other frameworks, we endorse the separation of the LLM's signature from its invocation. However, Cognify requires minimal change to your codebase. Defining a Cognify LM simply requires the following:
1. System prompt - we consider this to be the agent's role, necessary for optimizations like decomposition
2. Input variables - a placeholder for the actual user's request, necessary to construct high-quality few-shot examples

With Cognify, the above code looks like this:
```python
import cognify
from cognify.llm import InputVar, CogLM

system_prompt = "You are a helpful AI assistant built to answer questions."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

# define cognify agent
qa_question = InputVar(name="question")
cog_agent = CogLM(agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question]
)

@cognify.register
def call_qa_llm(question):
    messages = [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": f"{question}",
      }
    ]
    return cog_agent.forward(
      messages, 
      model_kwargs, 
      inputs={qa_question: question}
    )
```

Upon creation of a `CogLM` instance, it is automatically registered with the optimizer. Therefore, in order to have a consistent set of optimization targets, all `CogLM`s should be initialized globally. After that, the API call is simply replaced with the invocation of `CogLM.forward`. Under the hood, we use `litellm` to invoke the user's request. 


## Structured Output

We also support structured output. Original code:
```python
import openai
from pydantic import BaseModel

system_prompt = "You are a helpful AI assistant built to answer questions."

class Response(BaseModel):
  answer: str

model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

def call_qa_llm(question):
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": f"{question}"
    }
  ]

- return openai.chat.completions.parse(
    messages=messages,
    response_format=Response,
    **model_kwargs
  )
```

Ours:
```python
import cognify
from cognify.llm import InputVar, StructuredCogLM, OutputFormat
from pydantic import BaseModel

system_prompt = "You are a helpful AI assistant built to answer questions."

class Response(BaseModel):
  answer: str

qa_question = InputVar(name="question")
struct_cog_agent = StructuredCogLM(
  agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question],
  output_format=OutputFormat(schema=Response)
)

def call_qa_llm(question):
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": f"{question}"
    }
  ]

  return struct_cog_agent.forward(
    messages, 
    inputs={qa_question: question}
    model_kwargs, 
  )
```

## Image Inputs

You can also specify image inputs. Original code:

```python
import openai

system_prompt = "You are a helpful AI assistant built to answer questions about an image."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

def call_qa_llm(question, image_path):
    messages = [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": f"{question}"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{image_path}"
            }
          }
        ]
      },
    ]
    return openai.chat.completions.create({
      messages: messages,
      model_kwargs: model_kwargs
    })
```

Ours:

```python
import cognify
from cognify.llm import InputVar, ImageParams, CogLM

system_prompt = "You are a helpful AI assistant built to answer questions about an image."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

qa_question = InputVar(name="question")
qa_image = InputVar(name="image", ImageParams(is_image_upload=True, file_type='png'))
cog_agent = CogLM(agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question, qa_image]
)

@cognify.register
def call_qa_llm(question, image_path):
    messages = [
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": f"{question}"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/png;base64,{image_path}"
            }
          }
        ]
      },
    ]
    return cog_agent.forward(
      messages, 
      model_kwargs, 
      inputs={qa_question: question, qa_image: image_path}
    )
```