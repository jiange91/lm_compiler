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
from langchain_openai import ChatOpenAI

from compiler.IR.program import Workflow, Module, StatePool, Branch, Input, Output, hint_possible_destinations, Identity
from compiler.IR.llm import LMConfig, LLMPredictor, LMSemantic
from compiler.IR.base import ComposibleModuleInterface
from compiler.IR.rewriter.utils import add_argument_to_position, RewriteBranch
from compiler.IR.schema_parser import json_schema_to_pydantic_model
from compiler.IR.modules import CodeBox
from compiler.optimizer.prompts import *
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

logger = logging.getLogger(__name__)

class GraphInliner:
    """Inline inner modules of all immediate sub-graphes
    
    If need to inline deepest composible modules, need to recursively call this
    """
    def __init__(
        self,
        workflow: Workflow,
    ) -> None:
        self.workflow = workflow
        self.sub_graphs = workflow.get_all_modules(lambda x: isinstance(x, Workflow))
    
    def inline_sub_graph(self, sub_graph: Workflow):
        # migrate all inner modules to the parent workflow
        inner_modules = sub_graph.immediate_submodules()
        
        for module in inner_modules:
            # TODO: maybe apply name conversion to avoid collision
            self.workflow.add_module(module, reset_parent=True)
        
        # inline static dependencies
        self.workflow.static_dependencies.update(sub_graph.static_dependencies)
        # inline branches
        self.workflow.branches.update(sub_graph.branches)
        
        """pass through sub-graph boundries to input/output directly
        """
        # pass-through sub-graph boundary
        self.workflow.replace_node(sub_graph, sub_graph.start, sub_graph.end)
        
        sub_start_id = Identity(f'{sub_graph.name}_start_id')
        self.workflow.add_module(sub_start_id)
        sub_end_id = Identity(f'{sub_graph.name}_end_id')
        self.workflow.add_module(sub_end_id)
        self.workflow.replace_node(sub_graph.start, sub_start_id, sub_start_id)
        self.workflow.replace_node(sub_graph.end, sub_end_id, sub_end_id)
        
        # pass-through input/output nodes
        # TODO: finish this
        
        logger.info(f'Inlining sub-graph {sub_graph.name} Succeed')
        
        
    def flatten(self):
        for sub_graph in self.sub_graphs:
            self.inline_sub_graph(sub_graph)
        self.workflow.compile()
        logger.info('Flatten Workflow Succeed')
        self.workflow.visualize('examples/langraph_rag_src/flatten_workflow_viz')