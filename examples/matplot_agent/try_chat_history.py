from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from compiler.utils import load_api_key, get_bill

load_api_key('secrets.toml')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{name}: {input}"),
    ]
)
runnable = prompt | model

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

single_chat_hist = ChatMessageHistory()

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

def get_session_history() -> BaseChatMessageHistory:
    return single_chat_hist


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

print(with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?", "name": "Alice"},
    # config={"configurable": {"session_id": "abc123"}},
))


print(with_message_history.invoke(
    {"ability": "math", "input": "can you repeat Alice's question ?", "name": "Bob"},
))