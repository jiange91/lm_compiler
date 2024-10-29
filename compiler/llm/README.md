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
1. System prompt - we consider this to be the agent's role
2. Input variables - a placeholder for the actual user's request

With Cognify, the above code looks like this:
```python
from cognify.llm import InputVar, CogLM

system_prompt = "You are a helpful AI assistant built to answer questions."
model_kwargs = {'model': 'gpt-4o-mini', 'temperature': 0.0, 'max_tokens': 100}

# define cognify agent
qa_question = InputVar(name="question")
cog_agent = CogLM(agent_name="qa_agent",
  system_prompt=system_prompt,
  input_variables=[qa_question]
)

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

As you can see, most of the original code is untouched. Our optimization techniques function over a lightweight wrapper on top of the endpoint. Under the hood, we use `litellm` to invoke the user's request. 



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
from cognify.llm import InputVar, StructuredCogLM, OutputFormat
from pydantic import BaseModel

system_prompt = "You are a helpful AI assistant built to answer questions."

class Response(BaseModel):
  answer: str

# define structured cognify agent
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
