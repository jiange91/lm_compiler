from compiler.utils import load_api_key, get_bill
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda


import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

from langchain_core.messages import AIMessage, HumanMessage

human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(
    content="""\
1. Choose a programming language: Decide on a programming language that you want to learn.

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
"""
)


from langchain_openai.chat_models import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

add_on = HumanMessage("Please do not overlook any important details.")
chat_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="conversation"), 
        human_message_template, 
        add_on
    ]
).partial(not_related='not related')

def inspect_merge(inputs, **kwargs):
    result = merge_message_runs(inputs, **kwargs)
    print(result)
    return result

runnable = RunnableLambda(inspect_merge)

runnable = chat_prompt | inspect_merge | model
print(runnable.invoke(
    {"conversation": [human_message, ai_message], 
     "word_count": "10"}))
# exit()

from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic
from compiler.IR.program import Workflow, Context, hint_possible_destinations, StatePool
from compiler.IR.modules import Input, Output, CodeBox


semantic = LangChainSemantic(
    system_prompt="",
    inputs=["word_count", "conversation"],
    output_format="answer",
    following_messages=[
        MessagesPlaceholder(variable_name="conversation"), 
        human_message_template, 
        add_on,
    ]
)

qa_agent = LangChainLM('qa_agent', semantic)

qa_flow = Workflow('qa_flow')
qa_flow.add_module(Input('start'))
qa_flow.add_module(Output('end'))
qa_flow.add_module(qa_agent)

qa_flow.add_edge('start', 'qa_agent')
qa_flow.add_edge('qa_agent', 'end')
qa_flow.compile()

state = StatePool()
state.init({
    "conversation": [human_message, ai_message], 
    "word_count": "10"
})
qa_flow.pregel_run(state)

print(state.news('answer'))