from langchain_core.messages.utils import get_buffer_string
from langchain_core.messages.base import BaseMessage

# NOTE: input should be easily converted to string
#   as these variables will be directly passed to the model invocation
def var_2_str(var):
    if isinstance(var, str):
        return var
    if isinstance(var, list):
        if var and isinstance(var[0], BaseMessage):
            return get_buffer_string(var)
    return str(var)