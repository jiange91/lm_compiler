# LangChain

In Langchain, the `Runnable` class is the primary abstraction for executing a task. To create a `cognify.Model` from a runnable chain, the chain must contain a chat prompt template, a chat model, and optionally an ouptut parser. The chat prompt template is used to construct the system prompt and obtain the input variables, the chat model is used to obtain the language model config, and the output parser is used to construct the output format. If no output parser is provided, Cognify will assign a default label. 

The translatable runnables should follow the following formats:
- `BaseChatPromptTemplate | BaseChatModel`
- `BaseChatPromptTemplate | BaseChatModel | BaseOutputParser`

The purpose of this is to separate the definition of the runnable from its invocation. If there are `RunnableLambda`s interspersed, changes to any prompt templates or model outputs only take place at runtime and can differ based on the input. Once a runnable has been translated, it can be freely used in more complex chains. 

For more control over which runnables are optimized, pass the `--no-translate` flag to the `$ cognify optimize` command. Then, manually connect a Langchain `Runnable` to Cognify by wrapping your `Runnable` with our wrapper class `cognify.RunnableModel`:
```python
from langchain_core._prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class CapitalCity(BaseModel):
  confidence: float
  city: str

prompt = ChatPromptTemplate([
  ("system", "You are a helpful AI bot."),
  ("human", "What is the capital of {country}?")
])
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
parser = PydanticOutputParser(pydantic_object=CapitalCity)
runnable = prompt | model | parser

### wrap your runnable like so
import cognify
runnable = cognify.RunnableModel(runnable, name="gen_capital_city")

def call_capital_city_qa(country):
  runnable.invoke({"country": country}) ### invocation code remains unchanged
```

Additionally, all runnables that you intend to optimize (your optimization targets) should be initialized globally to allow the optimizer to hook onto them. If your runnable is initialized within a function that also handles its invocation, you should extract the definition and initialize a `cognify.RunnableModel` like the code shown above. The `cognify.RunnableModel` instance can then be freely used anywhere in your code.

## LangGraph

LangGraph is an orchestrator that is agnostic to the underlying framework. It can be used to orchestrate Langchain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our `@cognify.workflow_entry` decorator wherever you are invoking your compiled graph like so: 

```python
# adapted from https://github.com/langchain-ai/langgraph-example/blob/main/my_agent/agent.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node
from my_agent.utils.state import AgentState

...

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

...

workflow.add_edge("action", "agent")

graph = workflow.compile()

# -- your invocation code + decorator --
import cognify
@cognify.workflow_entry
def call_langgraph(prompt: str):
  graph.invoke({"messages": [HumanMessage(content=f"{prompt}")]})
```