from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import get_buffer_string

messages = [
    HumanMessage(content="Hi, how are you?"),
    AIMessage(content="Good, how are you?"),
]
print(get_buffer_string(messages))
# -> "Human: Hi, how are you?