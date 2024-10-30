import os

os.environ['DSP_CACHEBOOL'] = 'false'
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from compiler.utils import load_api_key
from dspy.functional import TypedPredictor
from compiler.llm.connectors.dspy import PredictCogLM

load_api_key('/mnt/ssd4/lm_compiler/secrets.toml')

gpt4o_mini = dspy.LM('gpt-4o-mini', max_tokens=1000)

colbert = dspy.ColBERTv2(url='http://192.168.1.16:8893/api/search')

dspy.configure(lm=gpt4o_mini, rm=colbert)

from dsp.utils.utils import deduplicate

class BasicMH(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
        self.generate_answer = PredictCogLM(dspy.ChainOfThought("context, question -> answer"), name="generate_answer")

        print(type(self.generate_answer))


    def forward(self, question):
        context = []

        for hop in range(2):
            search_query = self.generate_query[hop](context=context, question=question).search_query
            print("Search query:", search_query)
            passages = self.retrieve(search_query).passages
            print("Passages:", passages)
            context = deduplicate(context + passages)

        answer = self.generate_answer(context=context, question=question).copy(context=context)
        dspy.Prediction
        return answer
    
agent = BasicMH(passages_per_hop=2)


from dspy.adapters.chat_adapter import ChatAdapter
from pprint import pprint

adapter = ChatAdapter()

print(agent.named_predictors())


from pydantic import BaseModel, create_model
from typing import Any, Dict, Type

def generate_pydantic_model(model_name: str, fields: Dict[str, Type[Any]]) -> Type[BaseModel]:
    # Generate a dynamic Pydantic model using create_model
    return create_model(model_name, **{name: (field_type, ...) for name, field_type in fields.items()})

new_agent = agent.deepcopy()

for predictor in agent.predictors():
    print(predictor.__class__.__name__)
    print(predictor.extended_signature.input_fields.keys())
    print(predictor.extended_signature.output_fields.items())
    output_fields = {k: v.annotation for k, v in predictor.extended_signature.output_fields.items()}
    print(output_fields)

    # # Create the Pydantic model
    # PersonModel = generate_pydantic_model('PersonModel', output_fields)

    # # Instantiate and validate data with the model
    # person_instance = PersonModel(reasoning="hello", search_query="goodbye")

    # print(isinstance(person_instance, BaseModel))  # Output: True
    # print(person_instance.reasoning)
    messages = adapter.format(predictor.extended_signature, predictor.demos, {"context": "my context", "question": "my question"})
    #pprint(messages)
    
    
    # print(predictor.signature)
    # print(predictor.demos)



print(dspy.settings)