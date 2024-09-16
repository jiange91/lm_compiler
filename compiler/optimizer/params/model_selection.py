from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor 

class LMSelection(ParamBase):
    level = ParamLevel.NODE
    
    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = data['name'], data['module_name'], data['default_option'], data['options']
        options = [ModelOption(dat['model']) for name, dat in options.items()]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )
    
    
class ModelOption(OptionBase):
    def __init__(self, model: str):
        super().__init__(model)
        self.model = model
        
    def apply(self, lm_module: LLMPredictor):
        lm_module.lm_config['model'] = self.model
        # model selection will take effect after reset and set
        # which is performed at the optimizer side
        return lm_module
    
    def to_dict(self):
        base = super().to_dict()
        base['model'] = self.model
        return base

def model_option_factory(models: list[str]):
    return [ModelOption(model) for model in models]