from compiler.IR.modules import LLMPredictor, LMConfig
from langchain_openai import ChatOpenAI
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.callbacks import BaseCallbackHandler
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class LLMTracker(BaseCallbackHandler):
    def __init__(self, cmodule: 'LangChainLM'):
        super().__init__()
        self.cmodule = cmodule
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        meta = response.llm_output['token_usage']
        meta['model'] = response.llm_output['model_name']
        self.cmodule.llm_gen_meta.append(deepcopy(meta))

class LangChainLM(LLMPredictor):
    def __init__(self, name, kernel) -> None:
        super().__init__(name, kernel)
        self.llm_gen_meta = []
    
    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        model_name: str = self.lm_config['model']
        if model_name.startswith('gpt-'):
            self.lm = ChatOpenAI(**self.lm_config, callbacks=[LLMTracker(self)])
        else:
            raise ValueError(f"Model {model_name} not supported")
        self.kernel.lm = self.lm
        return 

    def get_lm_history(self):
        return self.llm_gen_meta[-1]