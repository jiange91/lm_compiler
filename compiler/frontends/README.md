# Frontends

We currently support connectors from Langchain and DSPy. This means if you use either of those frameworks to define your gen-AI workflow, it can be easily translated into our representation with little to no changes to your code. For any framework, we preserve the original program semantics as much as possible (see: [DSPy]()).

## Langchain

Connecting a Langchain `Runnable` to Cognify simply involves wrapping your `Runnable` with our wrapper class `RunnableCogLM` like so:
```python
from langchain_core._prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from cognify.frontends.langchain import RunnableCogLM

class CapitalCity(BaseModel):
  city: str

prompt = ChatPromptTemplate([
  ("system", "You are a helpful AI bot."),
  ("human", "What is the capital of {country}?")
])
model = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
parser = PydanticOutputParser(pydantic_object=CapitalCity)
runnable = prompt | model | parser
runnable = RunnableCogLM(runnable, name="generate_capital") # wrap with cogLM and optional name field

def call_capital_city_qa(country):
  runnable.invoke({"country": country})
```

Much like with the OpenAI compatible API, the invocation code remains the same.

The translatable runnables should follow the following formats:
- BaseChatPromptTemplate | BaseChatOpenAI
- BaseChatPromptTemplate | BaseChatOpenAI | BaseOutputParser

The purpose of this is to separate the definition of the runnable from its invocation. If there are `RunnableLambda`s interspersed, changes to any prompt templates or model outputs only take place at runtime and can differ based on the input.

Additionally, all runnables that you intend to optimize (your optimization targets) should be initialized globally to allow the optimizer to hook onto them. If your runnable is initialized within a function that also handles its invocation, you should extract the definition and initialize a `RunnableCogLM` like the code shown above. The `RunnableCogLM` instance can then be freely used anywhere in your code. You can use our [translation tool]() to assist with this process.  

## DSPy

Connecting a DSPy `Predict` module to Cognify simply involves wrapping the module with our wrapper class `PredictCogLM` like so:
```python
import dspy
from cognify.frontends.dspy import PredictCogLM

# ... setup dspy lm and retriver ...

class SingleHop(dspy.Module):
  def __init__(self):
    self.retrieve = dspy.Retrieve(k=3)
    self.generate_answer = PredictCogLM(
      dspy.Predict("context,question->answer"),
      name="rag_qa"
    ) # wrap with cogLM and optional name field
  
  def forward(self, question):
    context = self.retrieve(question).passages
    answer = self.generate_answer(context=context, question=question)
    return dspy.Prediction(context=context, answer=answer)
```

DSPy is a tool that automatically generates prompts on behalf of the user, which we access directly at the message passing layer. However, there is one caveat: reasoning. A core difference between Cognify and DSPy is we require users to specify reasoning as a cog at the optimizer level, while they require users to specify reasoning at the programming level. Because the optimizer interacts with the user's prompts, we strip explicit reasoning prefixes from the predictor.

By default, DSPy provides structured output back to the user in the form of a `Prediction`. We preserve this behavior so `forward()` can remain unchanged. 

# Translation

The translation CLI interface operates on a set of files. We do this because it's likely your project extends beyond the scope of a single file. Instead of cycling through imports from a single entry point, which may result in the tool translating more files than necessary, we translate each file that is specified on the command line independently. 
```
cognify translate --files /path/to/my_workflow.py /path/to/my_agents.py --frontend langchain|dspy
```
**Note:** translation will generate a new file and a backup copy of your old code. As with any code modification tools, you should always inspect the output of the translation to ensure it is consistent with your expectation. Translation currently operates indiscriminately on all runnables/modules in a file. If you wish to exclude certain ones from the optimization, then you should either refrain from using the translation tool or manually modify the translated file. 

## Langchain

LangChain does not inherently differentiate between a signature/invocation. Our translation tool makes a best-effort refactor of your code to extract the runnable into a function that returns a `RunnableCogLM`, which is assigned to a global variable that replaces the previous local runnable. This ensures that the `RunnableCogLM` is registered at program initialization. 

## DSPy

DSPy separates the signature of a module from its invocation. This allows our translation tool to insert a `PredictCogLM` wrapper inside your main workflow module's `__init__()` without refactoring any code. At runtime, any modules that are not LM-based will be ignored since Cognify's optimization currently only targets language models.

## Langgraph
LangGraph is a tool made by the creators of LangChain that offers orchestration. However, it is agnostic to the underlying framework. Hence, translating files that contain LangGraph code is the same as attempting to translate a file that uses either of our supported underlying frameworks.

