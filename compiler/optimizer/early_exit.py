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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from compiler.IR.program import Workflow, Module, StatePool, Branch, Input, Output, hint_possible_destinations
from compiler.IR.llm import LMConfig, LLMPredictor, LMSemantic
from compiler.IR.rewriter.utils import add_argument_to_position, RewriteBranchReturn
from compiler.llm.schema_parser import json_schema_to_pydantic_model
from compiler.IR.modules import CodeBox
from compiler.optimizer.prompts import *
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

logger = logging.getLogger(__name__)

class CycleEarlyExist:
    def __init__(
        self,
        workflow: Workflow,
        module_2_options: Union[dict[str, list[str]], list[str]],
        trainset_input: Iterable[StatePool],
        trainset_label: Iterable[Any],
    ):
        self.workflow = workflow
        
        if not isinstance(module_2_options, dict):
            module_2_options = {m.name: module_2_options for m in self.lm_modules}
        for lm in self.workflow.get_all_modules(lambda x: isinstance(x, LLMPredictor)):
            lm.lm_config['model'] = module_2_options[lm.name]
        
        self.trainset_input = trainset_input
        self.trainset_label = trainset_label
        
        # Get labels of all branches
        
    def add_exit(
        self,
    ):
        pass