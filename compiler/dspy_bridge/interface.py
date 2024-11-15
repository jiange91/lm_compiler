from typing import Dict, Union
import dspy
import json
from dataclasses import dataclass, field
import logging

from compiler.IR.modules import LLMPredictor, LMConfig

logger = logging.getLogger(__name__)



def dump_lm_history(lm, path):
    with open(path, 'w+') as f:
        json.dump(lm.history, f)

def log_after_generation(instance):
    original_call = instance.forward
    
    def new_call(*args, **kwargs):
        result = original_call(*args, **kwargs)
        dump_lm_history(instance._predict.lm, f"{getattr(instance, '_compiler_name')}_lm_hist.json")
        return result
    instance.forward = new_call
    
def dspy_adaptor(m: dspy.Module):
    m._include_in_optimizer = True
    return m


class DSPyLMConfig(LMConfig):
    def to_json(self):
        return json.dumps(self.kwargs)
    
    def from_json(self, data):
        self.kwargs = json.loads(data)
    

class DSPyLM(LLMPredictor):
    
    def set_lm(self):
        self.lm = dspy.OpenAI(**self.lm_config)
        logger.debug(f'Setting LM for {self.name}: {self.lm}')
    
    def get_lm_history(self):
        metas = []
        for history in self.lm.history:
            meta = {
                'prompt_tokens': history['response']['usage']['prompt_tokens'],
                'completion_tokens': history['response']['usage']['completion_tokens'],
                'model': history['response']['model']
            }
            metas.append(meta)
        return metas
    
    # NOTE: deprecated, not compatible to Dspy thread management
    def wrap_kernel_with_context(self, kernel):
        class KWrapper:
            def __init__(self, kernel, parent):
                self.kernel = kernel
                self.parent = parent
            def __call__(self, *args, **kwargs):
                with dspy.settings.context(lm=self.parent.lm):
                    return self.kernel(*args, **kwargs)
                
        return KWrapper(kernel, self)