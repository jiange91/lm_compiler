import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal
import copy
import logging
import numpy as np
import inspect
from collections import defaultdict

from langchain_core.output_parsers import NumberedListOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from compiler.IR.program import Workflow, Module, StatePool
from compiler.IR.llm import LMConfig, LLMPredictor, AgentDecomposeMeta
from compiler.optimizer.utils import json_format_instructions

logger = logging.getLogger(__name__)

# ================== Complexity Estimation ==================

class ComplexityEstimation(BaseModel):
    """complexity of each agent"""
    score: int = Field(
        description="complexity score of the agent"
    )
    rationale: str = Field(
        description="rationale for the complexity score"
    )

class ComplexityList(BaseModel):
    """complexity of all agents"""
    es: list[ComplexityEstimation] = Field(
        description="complexity of all agents"
    )


complexity_system = """
You are an expert at designing LLM-based agent workflow. Your task is to evaluate the complexity of the responsibility of each agent. 

You will be provided with their system prompts for reference. Please assign each agent a numerical rating on the following scale to indicate the complexity. You should consider:
Does the task require multi-step reasoning, planning, decision-making? 
Does the task encompass a wide range of responsibilities?
For each agent, please give your rate from 1 to 5 and provide your rationale for the rating.

Rating criteria: 1: straightforward, 5: very complex. 

Please follow the order of the given agents.

{output_format}
"""

complexity_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", complexity_system),
        ("human", "Agent prompts: {agents}\n\nComplexity Analysis: \n\n"),
    ]
).partial(output_format=json_format_instructions(ComplexityList.schema()))

def estimate_complexity_kernel(agents: list[str]):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.75).with_structured_output(ComplexityList)
    routine = complexity_prompt | llm
    agent_prompts = []
    for i, agent in enumerate(agents):
        agent_prompts.append(f"Prompt {i+1}:\n {agent}")
    complexity = routine.invoke({"agents": '\n'.join(agent_prompts)}).es
    output = []
    for e in complexity:
        output.append((e.score, e.rationale))
    return output


# ================== High-level Decompose Task ==================

class Edge(BaseModel):
    """data dependency between agents"""
    src: str = Field(
        description="source agent"
    )
    dst: str = Field(
        description="destination agent"
    )

class DecomposeOutput(BaseModel):
    """return the refined workflow"""
    agent_prompts: list[str] = Field(
        description="a list of prompts for each agent in the refined workflow"
    )
    
    agent_dependencies: list[Edge] = Field(
        description="a list of data dependencies between agents"
    )

def from_output_to_meta(output: DecomposeOutput) -> AgentDecomposeMeta:
    deps = defaultdict(list)
    for e in output.agent_dependencies:
        deps[e.src].append(e.dst)
    return AgentDecomposeMeta(
        agent_prompts=output.agent_prompts,
        agent_dependencies=deps,
    )
    
decompose_system = """
You are an expert at designing LLM-based agent workflow. Your task is to design a set of agents to perform the given task, each with a clear and separate role. 

The given task is originally performed by a single LLM agent. You will be provided with its prompt for task description. Please pay attention to all the details in the prompt. Your should make sure that the new agent system should fulfill all the requirements in the original prompt. Also you will be provided with some rationale for the complexity of the task to help you decompose the task.

For the final output, you need to specify the prompt for each agent in your refined workflow and any dependencies between them.

Principles:
1. In all new prompts, please be clear about the role of the agent and provide detailed instruction to guide the agent to perform its task.
2. Decomposed agents should be independent and have clear responsibilities.
3. For each new agent, you should only use the information provided in the original prompt or the information generated by other new agents. Do not seek information from external sources like web-search or databases that do not exist in the orignal prompt.

{output_format}
"""

decompose_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", decompose_system),
        ("human", "Original Prompt:\n{prompt}\n\nComplexity Rationale:\n{rationale}\n\nYour answer:\n\n")
    ]
).partial(output_format=json_format_instructions(DecomposeOutput.schema()))

def decompose_kernel(task: str, complexity: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.75).with_structured_output(DecomposeOutput)
    routine = decompose_prompt | llm 
    new_workflow = routine.invoke({"prompt": task, "rationale": complexity})
    return new_workflow

    
# ================== Task Decompose Class ==================

class LMTaskDecompose:
    def __init__(
        self,
        workflow: Workflow,
    ):
        self.workflow = workflow
        self.lm_modules : list[LLMPredictor] = workflow.get_all_modules(lambda m: isinstance(m, LLMPredictor))
    
    def decompose(self, threshold: Literal[1, 2, 3, 4, 5] = 4):
        agent_prompts = [m.semantic.get_agent_role() for m in self.lm_modules]
        complexity = estimate_complexity_kernel(agent_prompts)
        
        decompose_candidates = [(lm, score, rationale) for lm, (score, rationale) in zip(self.lm_modules, complexity)]
        decompose_candidates = sorted(decompose_candidates, key=lambda x: x[1], reverse=True)
        
        for lm, score, rationale in decompose_candidates:
            logger.info(f"Complexity of {lm.name}: {score}\nrationale: {rationale}\n\n")
        
        decompose_worker = {}
        for lm, score, rationale in decompose_candidates:
            if int(score) >= threshold:
                logger.info(f"Decomposing prompt for module {lm.name}")
                new_workflow: DecomposeOutput = decompose_kernel(lm.semantic.get_agent_role(), rationale)
                logger.info("Decomposed agent: \n - " + "\n - ".join(new_workflow.agent_prompts))
                logger.info("Agent dependencies: \n - " + "\n - ".join(str(d) for d in new_workflow.agent_dependencies))
                decompose_worker[lm.name] = from_output_to_meta(new_workflow)
        
        for lm in self.lm_modules:
            if lm.name in decompose_worker:
                lm.semantic.decompose(decompose_worker[lm.name])