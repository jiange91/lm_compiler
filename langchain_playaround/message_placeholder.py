from compiler.utils import load_api_key, get_bill
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
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

# chat_prompt.format_prompt(
#     conversation=[human_message, ai_message], word_count="10"
# ).to_messages()


from langchain_openai.chat_models import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

runnable = chat_prompt | model
print(runnable.invoke(
    {"conversation": [human_message, ai_message], 
     "word_count": "10"}))


from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic
from compiler.IR.program import Workflow, Context, hint_possible_destinations, StatePool
from compiler.IR.modules import Input, Output, CodeBox

class AnswerSchema(BaseModel):
    """Response from QA"""
    answer: str = Field(
        description="The answer to the question"
    )

semantic = LangChainSemantic(
    system_prompt="",
    inputs=["word_count", "conversation"],
    output_format=AnswerSchema,
    enable_memory=True,
    input_key_in_mem='question'
)

qa_agent = LangChainLM('qa_agent', semantic)

qa_flow = Workflow('qa_flow')
qa_flow.add_module(Input('start'))
qa_flow.add_module(Output('end'))
qa_flow.add_module(qa_agent)

def next_question():
    return {"question": "Can you repeat last question asked ?"}
qa_flow.add_module(CodeBox('next_question', next_question))

@hint_possible_destinations(['next_question', 'end'])
def ask_or_not(ctx: Context):
    if ctx.invoke_time >= 1:
        return 'end'
    return 'next_question'
    

qa_flow.add_edge('start', 'qa_agent')
qa_flow.add_branch('ask or not', 'qa_agent', ask_or_not)
qa_flow.add_edge('next_question', 'qa_agent')
qa_flow.compile()

state = StatePool()
state.init({'question': 'What does cosine mean?'})
qa_flow.pregel_run(state)

print(state.all_history(fields=['question', 'answer']))