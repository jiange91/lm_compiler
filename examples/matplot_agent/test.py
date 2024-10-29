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

from compiler.IR.modules import CodeBox
from compiler.optimizer.prompts import *
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.llm import LMConfig

logger = logging.getLogger(__name__)

load_api_key('secrets.toml')

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = load_image("/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/sample_runs_direct_expend_cot/6_546b7c2e7bce485da4e17a02c36ec75c/novice.png")

dummy_prmopt = """
You will be given a data visualization from user. Please describe in detail what you see in the plot and what you think the user wants to see in the plot.
"""

visual_refinement_semantic = LangChainSemantic(
    dummy_prmopt,
    # ['query', 'code', 'plot_image'],
    ['plot_image'],
    # "visual_refinement",
    "answer",
    img_input_idices=[0],
)
visual_refinement_lm = LangChainLM('visual_refinement', visual_refinement_semantic, opt_register=True)
visual_refine_lm_config = LMConfig(
    provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)
visual_refinement_lm.lm_config = visual_refine_lm_config
visual_refinement_agent = visual_refinement_lm.as_runnable()


answer = visual_refinement_agent.invoke({'plot_image': base64_image})

print(answer)
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