import os
import json
from typing import Union, Optional, Any, Tuple, Callable, Iterable, Literal, Dict, List
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
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from compiler.IR.program import Workflow, Module, StatePool, Branch, Input, Output, hint_possible_destinations
from compiler.IR.llm import LMConfig, LLMPredictor, LMSemantic
from compiler.IR.rewriter.utils import add_argument_to_position, RewriteBranch
from compiler.IR.schema_parser import json_schema_to_pydantic_model, get_pydantic_format_instruction
from compiler.IR.modules import CodeBox
from compiler.optimizer.prompts import *
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

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
    es: List[ComplexityEstimation] = Field(
        description="list of complexity descriptions"
    )
    

def estimate_complexity_kernel(agents: list[str]):
    complexity_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", complexity_system),
            ("human", "Here are the agent prompts: {agents}\n\nNow please give your Complexity Analysis.\nPlease follow the format instructions:\n{format_instructions}\n\n"),
        ]
    ).partial(format_instructions=get_pydantic_format_instruction(ComplexityList))
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=ComplexityList)
    routine = complexity_prompt | llm | parser
    agent_prompts = []
    for i, agent in enumerate(agents):
        agent_prompts.append(f"Prompt {i+1}:\n {agent}")
    complexity = routine.invoke({"agents": '\n'.join(agent_prompts)})
    complexity = ComplexityList.model_validate(complexity).es
    output = []
    for e in complexity:
        output.append((e.score, e.rationale))
    return output


# ================== High-level Decompose Task ==================

class AgentPropose(BaseModel):
    """Proposed agent information"""
    name: str = Field(
        description="name of the agent"
    )
    prompt: str = Field(
        description="prompt for the agent"
    )

class HighLevelDecompose(BaseModel):
    """High level decomposition of the task"""
    agents: List[AgentPropose] = Field(
        description="list of proposed agents"
    )

def high_level_decompose_kernel(task: str, complexity: str):
    decompose_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", decompose_system),
            ("human", "Original Prompt:\n{prompt}\n\nComplexity Rationale:\n{rationale}\n\nNow please give your solution with the following format:\n{format_instructions}\n\n")
        ]
    ).partial(format_instructions=get_pydantic_format_instruction(HighLevelDecompose))
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=HighLevelDecompose)
    routine = decompose_prompt | llm | parser
    new_agents = routine.invoke({"prompt": task, "rationale": complexity})
    new_agents = HighLevelDecompose.model_validate(new_agents)
    return new_agents

# ================== Refine New Agent Workflow ==================

class AgentMeta(BaseModel):
    """Information about each agent"""
    inputs: List[str] = Field(
        description="list of inputs for the agent"
    )
    outputs: List[str] = Field(
        description="list of outputs for the agent"
    )
    prompt: str = Field(
        description="refined prompt for the agent"
    )
    next_action: List[str] = Field(
        description="all possible next agents to invoke"
    )
    dynamic_action_decision: str = Field(
        "python code for dynamically deciding the next action, put 'None' if not needed"
    )
    
class NewAgentSystem(BaseModel):
    """New agent system"""
    agents: Dict[str, AgentMeta] = Field(
        description="dictionary of agent name to information about that agent"
    )
    

user_prompt = """

Now, this is the real user question for you:

## Information of the existing agent system
{old_info}

## Suggested new agents, with name and prompt
{new_agents}

Your answer:
"""


def decompose_refine_kernel(new_agent_name_prompt: dict[str, str], semantic: LMSemantic):
    decompose_refine_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", decompose_refine_system),
            ("human", user_prompt),
        ]
    ).partial(example_agent_info=example_agent_info,
              example_json_output=refine_example_json_output) # this is to avoid manual bracket escaping :)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0) # 4o-mini is not good
    routine = decompose_refine_prompt | llm | StrOutputParser()
    new_interaction = routine.invoke({
        "old_info": semantic.get_high_level_info(), 
        "new_agents": new_agent_name_prompt,
        }
    )
    decompose_refine_prompt.extend(
        [
            ("ai", "{new_interaction}"),
            ("human", "Please reformat your answer in the desired format.\n{format_instructions}"),
        ]
    )
    sllm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).with_structured_output(NewAgentSystem, method="json_mode")
    reformater = decompose_refine_prompt | sllm
    new_system = reformater.invoke({
        "old_info": semantic.get_high_level_info(), 
        "new_agents": new_agent_name_prompt,
        "new_interaction": new_interaction,
        "format_instructions": mid_level_system_format_instructions,
        }
    )
    return new_system

# ================== Finalize New Agents ==================
class AgentSemantic(BaseModel):
    """Information about each agent"""
    agent_prompt: str = Field(
        description="prompt for the agent"
    )
    inputs_variables: List[str] = Field(
        description="list of input variables for the agent"
    )
    output_json_schema: Union[Dict, str] = Field(
        description="json output schema for the agent, or the output variable name for simple text output"
    )
    next_action: List[str] = Field(
        description="all possible next agents to invoke"
    )
    dynamic_action_decision: str = Field(
        "python code for dynamically deciding the next action, put 'None' if not needed"
    )
    
class StructuredAgentSystem(BaseModel):
    """Refined agent system with structured output schema"""
    agents: Dict[str, AgentSemantic] = Field(
        description="dictionary of agent name to information about that agent"
    )
    
    final_output_aggregator_code: str = Field(
        description="python code to combine the outputs of the new agents to generate the final output, put 'None' if not needed"
    )

finalize_user_prompt = """

Now, this is the real task for you.

## Information of the old single-agent system
{old_semantic}

## Information of the suggested multi-agent system
{new_system}

## Your answer:
"""


def finalize_new_agents_kernel(old_semantic: LMSemantic, mid_level_desc: NewAgentSystem):
    # propose solution in pure text for strong reasoning
    interaction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{finalize_agent_system}"),
            ("human", finalize_user_prompt),
        ]
    ).partial(finalize_agent_system=finalize_new_agents_system)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    routine = interaction_prompt | llm | StrOutputParser()
    new_interaction = routine.invoke({
        "old_semantic": old_semantic.get_formatted_info(),
        "new_system": mid_level_desc.model_dump_json(indent=4),
        }
    )
    interaction_prompt.extend(
        [
            ("ai", "{new_interaction}"),
            ("human", "Now please reformat the new agent system to the desired JSON format.\n{format_instructions}"),
        ]
    )
    
    # refine the output to structured format
    sllm = ChatOpenAI(model="gpt-4o", temperature=0.0).with_structured_output(StructuredAgentSystem, method="json_mode")
    reformater = interaction_prompt | sllm
    soutput = reformater.invoke({
        "old_semantic": old_semantic.get_formatted_info(),
        "new_system": mid_level_desc.model_dump_json(indent=4),
        "new_interaction": new_interaction,
        "format_instructions": get_pydantic_format_instruction(StructuredAgentSystem),
        }
    )
    return soutput

# ================== Final aggregation code box ==================
def aggregator_factory(lm: LLMPredictor, code: str):
    old_output_schema = lm.semantic.get_output_schema()
    agg_func_obj = compile(code, '<string>', 'exec')
    local_name_space = {}
    exec(agg_func_obj, {}, local_name_space)
    func_name = agg_func_obj.co_names[0]
    aggregator = local_name_space[func_name]
    
    if old_output_schema is not None:
        @wraps(aggregator)
        def wrapper_kernel(**kwargs):
            result = aggregator(**kwargs)
            mresult = old_output_schema.model_validate(result)
            return {field: getattr(mresult, field) for field in lm.semantic.get_agent_outputs()}
    else:
        wrapper_kernel = aggregator       
    sig = inspect.signature(wrapper_kernel)
    logger.debug(f'Aggregator signature: {sig}')
    return wrapper_kernel
        
    
# ================== Task Decompose Class ==================
    
class LMTaskDecompose:
    def __init__(
        self,
        workflow: Optional[Workflow] = None,
        lm_modules: Optional[list[LLMPredictor]] = None,
    ):
        """provide either workflow or lm_modules to initialize the decomposer
        
        If both are provided, workflow will be used
        
        The provided workflow or lm_modules will be changed in-place
        """
        self.workflow = workflow
        if self.workflow is not None:
            self.lm_modules : list[LLMPredictor] = workflow.get_all_modules(lambda m: isinstance(m, LLMPredictor) and m.semantic.prompt_fully_manageable())
        else:
            if lm_modules is None:
                raise ValueError("Either workflow or lm_modules should be provided")
            self.lm_modules = lm_modules
        self.decompose_target_lms: list[LLMPredictor] = []
        
        # Cascading decomposition
        self.lm_2_new_agents: dict[str, HighLevelDecompose] = {}
        self.lm_2_new_system: dict[str, NewAgentSystem] = {}
        self.lm_2_final_system: dict[str, StructuredAgentSystem] = {}
    
    def prepare_decompose_metadata(
        self,
        target_modules,
        threshold,
        log_dir,
    ):
        log_path = os.path.join(log_dir, 'task_decompose_mid_level.json')
        # Get decomposition meta
        if os.path.exists(log_path):
            logger.info("mid-level decomposition already exists, read and skip sampling")
            with open(log_path) as f:
                json_lm_2_new_system = json.load(f)
                self.lm_2_new_system = {k: NewAgentSystem.model_validate(v) for k, v in json_lm_2_new_system.items()}
                self.decompose_target_lms = [
                    m for m in self.lm_modules 
                    if m.name in self.lm_2_new_system and (target_modules is None or m.name in target_modules)
                ]
        else:
            logger.info("Estimating complexity of agents")
            agent_prompts = [m.semantic.get_agent_role() for m in self.lm_modules if target_modules is None or m.name in target_modules]
            complexity = estimate_complexity_kernel(agent_prompts)
            
            decompose_candidates = [(lm, score, rationale) for lm, (score, rationale) in zip(self.lm_modules, complexity)]
            decompose_candidates = sorted(decompose_candidates, key=lambda x: x[1], reverse=True)
            
            for lm, score, rationale in decompose_candidates:
                logger.info(f"Complexity of {lm.name}: {score}\nrationale: {rationale}\n\n")
            with open(os.path.join(log_dir, 'task_decompose_candidates.json'), 'w+') as f:
                json.dump({lm.name: {'score': score, 'rationale': rationale} for lm, score, rationale in decompose_candidates}, f, indent=4)
            
            logger.info("Performing high-level agent decomposition")
            def _hd(lm, score, rationale):
                if int(score) >= threshold:
                    new_agents = high_level_decompose_kernel(lm.semantic.get_agent_role(), rationale)
                    self.lm_2_new_agents[lm.name] = new_agents
                    self.decompose_target_lms.append(lm)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda x: _hd(*x), decompose_candidates)
            
            logger.info('High-level decomposition results:\n')
            pprint(self.lm_2_new_agents)
            
            logger.info("Adding concrete dependencies to decomposed system")
            def _ld(lm: LLMPredictor):
                new_agents_name_prompt = {a.name: a.prompt for a in self.lm_2_new_agents[lm.name].agents}
                new_system = decompose_refine_kernel(new_agents_name_prompt, lm.semantic)
                self.lm_2_new_system[lm.name] = new_system
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_ld, lm) for lm in self.decompose_target_lms]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to decompose: {e}")
            # for lm in self.decompose_target_lms:
            #     _ld(lm)
            
            logger.info("Mid-level decomposition results:\n") 
            pprint(self.lm_2_new_system)
            json_lm_2_new_system = {k: json.loads(v.model_dump_json()) for k, v in self.lm_2_new_system.items()}
            with open(log_path, 'w+') as f:
                json.dump(json_lm_2_new_system, f, indent=4)
                
    def finalize_decomposition(self, target_modules, log_dir):
        log_path = os.path.join(log_dir, 'task_decompose_final.json')
        if os.path.exists(log_path):
            logger.info("final decomposition already exists, read and skip sampling")
            with open(log_path) as f:
                json_lm_2_final_system = json.load(f)
                self.lm_2_final_system = {k: StructuredAgentSystem.model_validate(v) for k, v in json_lm_2_final_system.items()}
                self.decompose_target_lms = [
                    m for m in self.lm_modules 
                    if m.name in self.lm_2_final_system and (target_modules is None or m.name in target_modules)
                ]
        else:
            logger.info("Finalizing new agent system")
            def _fd(lm: LLMPredictor):
                mid_level_desc: NewAgentSystem = self.lm_2_new_system[lm.name]
                final_agents = finalize_new_agents_kernel(lm.semantic, mid_level_desc)
                self.lm_2_final_system[lm.name] = final_agents
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_fd, lm) for lm in self.decompose_target_lms]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to finalize: {e}")
            # for lm in self.decompose_target_lms:
            #     _fd(lm)
            logger.info("Final decomposition results:\n")
            pprint(self.lm_2_final_system)
            with open(log_path, 'w+') as f:
                json_lm_2_final_system = {k: json.loads(v.json()) for k, v in self.lm_2_final_system.items()}
                json.dump(json_lm_2_final_system, f, indent=4)
    
    @staticmethod
    def materialize_decomposition(
        lm: LLMPredictor, 
        new_agents: StructuredAgentSystem, 
        default_lm_config: dict,
        log_dir: str,
    ) -> Workflow:
        """Actually transform the graph to apply the decomposition
        
        1. First create a sub-graph to represent the new agent system
        2. Then replace the original agent with the new agent system
            - If the encolsing module is a graph, flatten the sub-graph to avoid hierarchy
            - otherwise, replace the agent directly
        """
        # TODO: add more validate checks
        sub_graph = Workflow(f'{lm.name}_sub_graph')
        input_name, output_name = f'{lm.name}_sub_graph_input', f'{lm.name}_sub_graph_output'
        sub_graph.add_module(Input(input_name))
        sub_graph.add_module(Output(output_name))

        logical_end_name = output_name
        # Add final aggregator
        if new_agents.final_output_aggregator_code != 'None':
            code_kernel = aggregator_factory(lm, new_agents.final_output_aggregator_code)
            aggregator = CodeBox(f'{lm.name}_final_aggregator', code_kernel)
            sub_graph.add_module(aggregator)
            sub_graph.add_edge(aggregator.name, output_name)
            logical_end_name = aggregator.name
        
        # Materialize each agent
        name_2_new_lm: dict[str, LangChainLM] = {}
        img_input_names = set(lm.semantic.get_img_input_names())
        for agent_name, agent_meta in new_agents.agents.items():
            valid_file_name = agent_name.replace(" ", "").replace("\n", "").replace("\t", "") + '.py'
            module_fpath = os.path.join(log_dir, valid_file_name)
            if isinstance(agent_meta.output_json_schema, str):
                output_model = agent_meta.output_json_schema
            else:
                output_model = json_schema_to_pydantic_model(agent_meta.output_json_schema, module_fpath)
            # if only has one str output for no ending agents, remove output format
            if not isinstance(output_model, str) and 'END' not in agent_meta.next_action:
                if len(output_model.model_fields) == 1:
                    output_model = list(output_model.model_fields.keys())[0]
            # TODO: currently demo is ignored after decomposition
            if 'END' in agent_meta.next_action:
                need_type_hint, format_inst = lm.semantic.get_output_spec()
            else:
                need_type_hint, format_inst = True, None
            lm_semantic = LangChainSemantic(
                system_prompt=agent_meta.agent_prompt,
                inputs=agent_meta.inputs_variables,
                output_format=output_model,
                need_output_type_hint=need_type_hint,
                output_format_instructions=format_inst,
                img_input_idices=[i for i, name in enumerate(agent_meta.inputs_variables) if name in img_input_names],
            )
            agent_lm = LangChainLM(agent_name, lm_semantic)
            agent_lm.lm_config = default_lm_config if default_lm_config is not None else lm.lm_config
            name_2_new_lm[agent_name] = agent_lm
            sub_graph.add_module(agent_lm)
        
        # Get static dependency edges
        agent_2_srcs = defaultdict(list, {agent_name: [input_name] for agent_name in new_agents.agents}) # default to input node
        for agent_name, agent_meta in new_agents.agents.items():
            next_action = agent_meta.next_action
            is_static_edge = agent_meta.dynamic_action_decision == 'None'
            if is_static_edge:
                for next_agent in next_action:
                    if next_agent == 'END':
                        agent_2_srcs[logical_end_name].append(agent_name)
                    else:
                        agent_2_srcs[name_2_new_lm[next_agent].name].append(agent_name) # for check name existance
        for agent_name, srcs in agent_2_srcs.items():
            sub_graph.add_edge(srcs, agent_name)
            
        # Add dynamic dependency edges
        def get_branch_function(dynamic_action_code: str, next_actions: List[str]):
            next_actions_str = ', '.join([f'"{na}"' for na in next_actions])
            new_func, func_name = add_argument_to_position(dynamic_action_code, 'ctx', 0)
            template = f"""\
@hint_possible_destinations([{next_actions_str}])
{new_func}
"""
            clean_code = template.replace('END', f'{logical_end_name}')
            return clean_code, func_name
        
        for agent_name, agent_meta in new_agents.agents.items():
            next_action = agent_meta.next_action
            is_static_edge = agent_meta.dynamic_action_decision == 'None'
            if not is_static_edge:
                # dynamic edge
                decision_code = agent_meta.dynamic_action_decision
                clean_decision_code, func_name = get_branch_function(decision_code, next_action)
                func_obj = compile(clean_decision_code, '<string>', 'exec')
                local_name_space = {}
                exec(func_obj, {'hint_possible_destinations': hint_possible_destinations}, local_name_space)
                callable_code = local_name_space[func_name]
                sub_graph.add_branch(
                    f'condition_flow_after_{agent_name}', 
                    agent_name, 
                    callable_code, 
                    clean_decision_code,
                    enhance_existing=True,
                )
        return sub_graph
        
    def decompose(
        self,
        target_modules: Optional[list[str]] = None,
        log_dir: str = 'task_decompose_logs',
        threshold: Literal[1, 2, 3, 4, 5] = 4,
        default_lm_config: Optional[dict] = None,
        materialize: bool = True,
    ):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.prepare_decompose_metadata(target_modules, threshold, log_dir)
        self.finalize_decomposition(target_modules, log_dir)
        
        if materialize:
            for lm in self.decompose_target_lms:
                new_agent = self.materialize_decomposition(lm, self.lm_2_final_system[lm.name], default_lm_config, log_dir)
                # Replace the original agent with the sub-graph
                parent_node = lm.enclosing_module
                if not parent_node.replace_node(lm, new_agent, new_agent):
                    logger.error(f"Failed to replace {lm.name} with {new_agent.name}")
                else:
                    logger.info(f"Successfully replaced {lm.name} with {new_agent.name}")
            self.workflow.compile()
            self.workflow.visualize(os.path.join(log_dir, 'decomposed_workflow_viz'))
        