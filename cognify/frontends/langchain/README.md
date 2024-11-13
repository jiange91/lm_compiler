# LangChain

In Langchain, the `Runnable` class is the primary abstraction for executing a task. To create a `CogLM` from a runnable chain, the chain must contain a chat prompt template, a chat model, and optionally an ouptut parser. The chat prompt template is used to construct the system prompt and obtain the input variables, the chat model is used to obtain the language model config, and the output parser is used to construct the output format. If no output parser is provided, Cognify will assign a default label. 

The translatable runnables should follow the following formats:
- BaseChatPromptTemplate | BaseChatOpenAI
- BaseChatPromptTemplate | BaseChatOpenAI | BaseOutputParser

The purpose of this is to separate the definition of the runnable from its invocation. If there are `RunnableLambda`s interspersed, changes to any prompt templates or model outputs only take place at runtime and can differ based on the input. Once a runnable has been translated, it can be freely used in more complex chains. 

For more control over which runnables are optimized, pass the `--no-translate` flag to the `$ cognify optimize` command. Then, manually connect a Langchain `Runnable` to Cognify by wrapping your `Runnable` with our wrapper class `RunnableCogLM`:
```python
from langchain_core._prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from cognify.frontends.langchain import RunnableCogLM

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
runnable = RunnableCogLM(runnable, name="gen_capital_city") # wrap with cogLM and optional name field

def call_capital_city_qa(country):
  runnable.invoke({"country": country}) # invocation code remains unchanged
```

Additionally, all runnables that you intend to optimize (your optimization targets) should be initialized globally to allow the optimizer to hook onto them. If your runnable is initialized within a function that also handles its invocation, you should extract the definition and initialize a `RunnableCogLM` like the code shown above. The `RunnableCogLM` instance can then be freely used anywhere in your code.

## LangGraph

LangGraph is an orchestrator that is agnostic to the underlying framework. It can be used to orchestrate Langchain runnables, DSPy predictors, any other framework or even pure python. All you need to do to hook up your LangGraph code is use our `@cognify.workflow_entry` decorator wherever you are invoking your compiled graph like so: 

```python
# adapted from https://github.com/langchain-ai/langgraph-example/blob/main/my_agent/agent.py
import cognify
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Define the config
class GraphConfig(TypedDict):
  model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
  "agent",
  should_continue,
  {
    "continue": "action",
    "end": END,
  },
)

workflow.add_edge("action", "agent")

graph = workflow.compile()

# -- your invocation code + decorator --
@cognify.workflow_entry
def call_langgraph(prompt: str):
  graph.invoke({"messages": [HumanMessage(content=f"{prompt}")]})
```