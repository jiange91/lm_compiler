import autogen
from ..langchain_bridge.interface import LangChainLM, LangChainSemantic

class AutogenLM(autogen.ConversableAgent, LangChainLM):
    def __init__(self, name, llm_config, system_message=None, type="assistant", **kwargs):
        if type == "assistant":
            autogen.AssistantAgent.__init__(self, name=name, system_message=system_message, llm_config=llm_config, **kwargs)
        self.semantic =  LangChainSemantic(self, system_message,['task'])
        LangChainLM(name, self.semantic)
        self.langchain_formatted_messages = []
    
    def sync_messages(self):
        for message in self._oai_messages:
            formatted_message = {
                "type": "text",
                "content": str(message)
            }
        self.langchain_formatted_messages.append(formatted_message)

    def get_chat_history(self):
        # Return the transformed chat history
        return self.langchain_formatted_messages