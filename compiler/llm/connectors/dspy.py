import dspy
from dspy.adapters.chat_adapter import ChatAdapter, prepare_instructions
from compiler.llm import StructuredCogLM, InputVar, OutputFormat
from compiler.llm.model import LMConfig
import uuid
from pydantic import BaseModel, create_model
from typing import Any, Dict, Type
from litellm import ModelResponse

APICompatibleMessage = Dict[str, str] # {"role": "...", "content": "..."}

def generate_pydantic_model(model_name: str, fields: Dict[str, Type[Any]]) -> Type[BaseModel]:
    # Generate a dynamic Pydantic model using create_model
    return create_model(model_name, **{name: (field_type, ...) for name, field_type in fields.items()})

"""
Connector currently supports `Predict` with any signature and strips away all reasoning fields.
This is done because we handle reasoning via cogs for the optimizer instead of in a templated format. 
"""
class PredictCogLM(dspy.Module):
    def __init__(self, dspy_predictor: dspy.Predict, name: str = None):
        super().__init__()
        self.chat_adapter: ChatAdapter = ChatAdapter()
        self.predictor: dspy.Predict = dspy_predictor
        self.cog_lm: StructuredCogLM = self.cognify_predictor(dspy_predictor)
        self.output_schema = None

    def cognify_predictor(self, dspy_predictor: dspy.Predict, name: str = None) -> StructuredCogLM:
        # initialize cog lm
        agent_name = name or str(uuid.uuid4())
        system_prompt = prepare_instructions(dspy_predictor.extended_signature)
        input_names = list(dspy_predictor.extended_signature.input_fields.keys())
        input_variables = [InputVar(name=name) for name in input_names]

        output_fields = {k: v.annotation for k, v in dspy_predictor.extended_signature.output_fields.items()}
        self.output_schema = generate_pydantic_model("OutputData", output_fields)

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
        inputs: Dict[InputVar, str] = {k: kwargs[k.name] for k in self.cog_lm.input_variables}
        messages: APICompatibleMessage = self.chat_adapter.format(self.predictor.extended_signature,
                                                                self.predictor.demos,
                                                                inputs)
        response: ModelResponse = self.cog_lm.forward(messages, inputs) # model kwargs already set
        return self.cog_lm.output_format.schema.model_validate_json(response)