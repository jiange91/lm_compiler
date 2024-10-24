import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Dict, List
import copy
import logging
import numpy as np
import inspect
from collections import defaultdict
import concurrent.futures
from devtools import pprint
from functools import wraps, partial
from graphviz import Digraph
import base64

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from compiler.utils import load_api_key, get_bill

from compiler.IR.program import Workflow, Module, StatePool, Branch, Input, Output, hint_possible_destinations
from compiler.IR.llm import LMConfig, LLMPredictor, LMSemantic
from compiler.IR.rewriter.utils import add_argument_to_position, RewriteBranchReturn
from compiler.llm.schema_parser import json_schema_to_pydantic_model
from compiler.IR.modules import CodeBox
from compiler.optimizer.prompts import *
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

logger = logging.getLogger(__name__)

load_api_key('secrets.toml')

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = load_image("/mnt/ssd4/lm_compiler/examples/matplot_agent/test.png")

class ImgUnderstanding(BaseModel):
    """Response from the image understanding task"""
    answer: str = Field(
        description="The answer to the question about the image"
    )

img_understanding_semantic = LangChainSemantic(
    "Given an image, please answer the question about it",
    ['image_b64', 'question'],
    ImgUnderstanding,
    [0],
)

img_lm = LangChainLM('img understanding', img_understanding_semantic)
img_lm.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}

flow = Workflow('img understanding')
flow.add_module(Input('start'))
flow.add_module(Output('end'))
flow.add_module(img_lm)

flow.add_edge('start', 'img understanding')
flow.add_edge('img understanding', 'end')
flow.compile()

state = StatePool()
state.init({'image_b64': base64_image, 'question': 'What is the animal in this image?'})

flow.pregel_run(state)
print(state.news('answer'))
exit()


prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(
            template=[
                {
                    "type": "text",
                    "text": "What is the animal in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{base64_image}"
                    } 
                }
            ]
        )
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
routine = prompt | llm

print(routine.invoke({'base64_image': base64_image}))