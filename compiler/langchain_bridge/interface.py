from compiler.IR.modules import LLMPredictor, LMConfig
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class LangChainLM(LLMPredictor):
    def __init__(self, name, kernel) -> None:
        super().__init__(name, kernel)
    
    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        model_name: str = self.lm_config['model']
        if model_name.startswith('gpt-'):
            self.lm = ChatOpenAI(**self.lm_config)
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.kernel.lm = self.lm
        return 

    def get_lm_history(self):
        return None