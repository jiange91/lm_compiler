from compiler.optimizer.params.common import ParamBase, ParamLevel, OptionBase
from compiler.IR.llm import LLMPredictor 
from compiler.IR.llm import LMConfig
import uuid
import copy

class LMSelection(ParamBase):
    level = ParamLevel.NODE
    
    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = data['name'], data['module_name'], data['default_option'], data['options']
        options = [
            ModelOption(LMConfig.from_dict(dat['model_config']), tag) 
            for tag, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )

class ModelOption(OptionBase):
    def __init__(self, model_config: LMConfig, tag: str = None):
        # NOTE: this assumes provider + model is unique
        # use this as tag to increase config readability
        tag = tag or f'{model_config.provider}_{model_config.model}'
        super().__init__(tag)
        # NOTE: deepcopy is necessary in case module config is shared in memory
        self.model_config = copy.deepcopy(model_config)
    
    def _get_cost_indicator(self):
        return self.model_config.cost_indicator
       
    def apply(self, lm_module: LLMPredictor):
        lm_module.lm_config.update(self.model_config)
        # This is incase reset is not called, to trigger rebuild the kernel
        lm_module.lm = None
        return lm_module
    
    def to_dict(self):
        base = super().to_dict()
        base['model_config'] = self.model_config.to_dict()
        return base

def model_option_factory(model_configs: list[LMConfig]):
    return [ModelOption(cfg) for cfg in model_configs]