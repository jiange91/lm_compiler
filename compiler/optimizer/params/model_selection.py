from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor 

class LMSelection(ParamBase):
    level = ParamLevel.NODE
    
class ModelOption(OptionBase):
    def __init__(self, model: str):
        super().__init__(model)
        self.model = model
        
    def apply(self, lm_module: LLMPredictor):
        lm_module.lm_config['model'] = self.model
        # model selection will take effect after reset and set
        # which is performed at the optimizer side
        return lm_module

def model_option_factory(models: list[str]):
    return [ModelOption(model) for model in models]