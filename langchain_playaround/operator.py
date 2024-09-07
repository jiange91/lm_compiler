from operator import add

b = """
def langchain_lm_kernel():
{define_routine}
{invoke_routine}
    return
"""

a = """
    routine = self.semantic.chat_prompt_template | merge_message_runs() |  self.lm
    """

a_1 = """
    routine = RunnableWithMessageHistory(
        runnable=routine,
        get_session_history=lambda: self.chat_history,
        input_messages_key=self.semantic.input_key_in_mem,
        history_messages_key="compiler_chat_history",
    )
    """

c = """
    result = routine.invoke()
    """

b = b.format(define_routine=a+a_1, invoke_routine=c)

print(b)