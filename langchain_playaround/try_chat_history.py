from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from compiler.utils import load_api_key, get_bill
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json

load_api_key('secrets.toml')


class AnswerSchema(BaseModel):
    """Response from QA"""
    answer: str = Field(
        description="The answer to the question"
    )
    
parser = JsonOutputParser(pydantic_object=AnswerSchema)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer.\n{format_instructions}\n",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{name}: {input}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
runnable = prompt | model

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

single_chat_hist = ChatMessageHistory()


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history=lambda: single_chat_hist,
    # input_messages_key="input",
    history_messages_key="history",
)



structured = with_message_history | parser

#NOTE: langchain history only track one input ...
output = structured.invoke(
    {"ability": "math", "input": "What does cosine mean?", "name": "Alice"},
    # config={"configurable": {"session_id": "abc123"}},
)

# answer_1 = AnswerSchema.parse_obj(json.loads(output))
answer_1 = AnswerSchema.parse_obj(output)
print(answer_1)


print(with_message_history.invoke(
    {"ability": "math", "input": "Give the name and question in the previous round", "name": "Bob"},
).content)

print(single_chat_hist.messages)
# exit()

#==============
# Using our IR
#==============

from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic
from compiler.IR.program import Workflow, Context, hint_possible_destinations, StatePool
from compiler.IR.modules import Input, Output, CodeBox


semantic = LangChainSemantic(
    system_prompt="You are a QA system. Answer the question.",
    inputs=["question"],
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