import dspy
from dspy.adapters.chat_adapter import ChatAdapter, prepare_instructions
from compiler.llm import CogLM, StructuredCogLM, InputVar, OutputFormat
from compiler.llm.model import LMConfig
import uuid
from pydantic import BaseModel, create_model
from typing import Any, Dict, Type
from litellm import ModelResponse
import warnings

APICompatibleMessage = Dict[str, str] # {"role": "...", "content": "..."}

def generate_pydantic_model(model_name: str, fields: Dict[str, Type[Any]]) -> Type[BaseModel]:
    # Generate a dynamic Pydantic model using create_model
    return create_model(model_name, **{name: (field_type, ...) for name, field_type in fields.items()})

"""
Connector currently supports `Predict` with any signature and strips away all reasoning fields.
This is done because we handle reasoning via cogs for the optimizer instead of in a templated format. 
"""
class PredictCogLM(dspy.Module):
    def __init__(self, dspy_predictor: dspy.Module = None, name: str = None):
        super().__init__()
        self.chat_adapter: ChatAdapter = ChatAdapter()
        self.predictor: dspy.Module = dspy_predictor
        self.ignore_module = False
        self.cog_lm: StructuredCogLM = self.cognify_predictor(dspy_predictor)
        self.output_schema = None

    def cognify_predictor(self, dspy_predictor: dspy.Module = None, name: str = None) -> StructuredCogLM:
        if not dspy_predictor:
            return None
        
        if not isinstance(dspy_predictor, dspy.Predict):
            warnings.warn("Original module is not a `Predict`. This may result in lossy translation", UserWarning)
        
        if isinstance(dspy_predictor, dspy.Retrieve):
            self.ignore_module = True
            return None
            
        # initialize cog lm
        agent_name = name or str(uuid.uuid4())
        system_prompt = prepare_instructions(dspy_predictor.extended_signature)
        input_names = list(dspy_predictor.extended_signature.input_fields.keys())
        input_variables = [InputVar(name=name) for name in input_names]

        output_fields = dspy_predictor.extended_signature.output_fields
        if "reasoning" in output_fields:
            del output_fields["reasoning"]
            warnings.warn("Original module contained reasoning. This will be stripped. Add reasoning to the optimizer instead", UserWarning)
        output_fields_for_schema = {k: v.annotation for k, v in output_fields.items()}
        self.output_schema = generate_pydantic_model("OutputData", output_fields_for_schema)

        # lm config
        print(dspy.settings)
        lm_client: dspy.LM = dspy.settings.get('lm', None)
        assert lm_client, "Expected lm client, got none"
        lm_config = LMConfig(model=lm_client.model, kwargs=lm_client.kwargs)

        # always treat as structured to provide compatiblity with forward function 
        return StructuredCogLM(agent_name=agent_name,
                                system_prompt=system_prompt,
                                input_variables=input_variables,
                                output_format=OutputFormat(schema=self.output_schema),
                                lm_config=lm_config)

    def forward(self, **kwargs):
        assert self.cog_lm or self.predictor, "CogLM or Predictor must be initialized before invoking"

        if self.ignore_module:
            return self.predictor(**kwargs)
        else:
            inputs: Dict[InputVar, str] = {k: kwargs[k.name] for k in self.cog_lm.input_variables}
            messages: APICompatibleMessage = self.chat_adapter.format(self.predictor.extended_signature,
                                                                    self.predictor.demos,
                                                                    inputs)
            response: ModelResponse = self.cog_lm.forward(messages, inputs) # model kwargs already set
            kwargs: dict = self.cog_lm.output_format.schema.model_validate_json(response).model_dump()
            return dspy.Prediction(**kwargs)
        
def as_predict(cog_lm: CogLM) -> PredictCogLM:
    predictor = PredictCogLM(name=cog_lm.agent_name)
    if isinstance(cog_lm, StructuredCogLM):
        predictor.cog_lm = cog_lm
        predictor.output_schema = cog_lm.output_format.schema
    else:
        output_schema = generate_pydantic_model("OutputData", {cog_lm.get_output_label_name(): str})
        predictor.cog_lm = StructuredCogLM(agent_name=cog_lm.agent_name,
                                      system_prompt=cog_lm.system_prompt,
                                      input_variables=cog_lm.input_variables,
                                      output_format=OutputFormat(output_schema, 
                                                                 custom_output_format_instructions=cog_lm.get_custom_format_instructions_if_any()),
                                      lm_config=cog_lm.lm_config)
    return predictor
    