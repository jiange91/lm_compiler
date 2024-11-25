from typing import Union
import logging
import copy


logger = logging.getLogger(__name__)

from cognify.graph.base import Module, StatePool, ModuleStatus
from cognify.graph.program import Workflow, InputModule, Output
from cognify.graph.modules import CodeBox
from cognify.hub.cogs.common import (
    CogBase,
    CogLayerLevel,
    OptionBase,
    NoChange,
)
from cognify.llm import (
    Model,
    StructuredModel,
    StepInfo,
    Input,
    OutputFormat,
    OutputLabel,
)
from abc import ABCMeta


class ModuleEnsemble(CogBase):
    level = CogLayerLevel.GRAPH

    def __init__(
        self,
        options: list[OptionBase],
        name: str = "ensemble",
        default_option: Union[int, str] = 0,
        module_name: str = None,
        inherit: bool = True,
    ):
        super().__init__(name, options, default_option, module_name, inherit)

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )
        options = [
            EnsembleOptionMeta.registry[dat["type"]].from_dict(dat)
            for name, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class EnsembleOptionMeta(ABCMeta):
    registry: dict[str, type] = {"NoChange": NoChange}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls


class EnsembleOptionBase(OptionBase, metaclass=EnsembleOptionMeta):
    def __init__(
        self,
        name: str,
        num_path: int,
    ) -> None:
        super().__init__(name)
        self.num_path = num_path

    def _get_cost_indicator(self):
        return self.num_path + 1  # 1 for the aggregator

    def sample_then_aggregate(self, module: Module) -> Module:
        raise NotImplementedError

    def apply(self, module: Module) -> Module:
        ensemble = self.sample_then_aggregate(module)
        return ensemble


class SamplerPostProcess(CodeBox):
    def __init__(
        self,
        name: str,
        origin_expert: Model,
        experts: list[Model],
    ):
        super().__init__(name=name, kernel=None)
        # Added this incase we will apply prompt rewriting for experts
        self.origin_expert = origin_expert
        self.experts = experts

    @property
    def prefix(self):
        return f"{self.name}_question_context"

    def invoke(self, statep: StatePool):
        # get all context of the problem
        agent_task = self.origin_expert.get_agent_role()

        inputs_dict = {}
        inputs_dict = self.experts[0].steps[-1].filled_inputs_dict

        paths = []
        proposal_template = """
** Worker Proposal {i} **

Rationale: {rationale}

Answer: {response}
        """
        for i, expert in enumerate(self.experts):
            if expert.steps:
                step: StepInfo = expert.steps[-1]
                rationale = step.rationale
                output = step.output
                proposal = proposal_template.format(
                    i=i, rationale=rationale, response=output
                )
                paths.append(proposal)
        paths_str = "\n---\n".join(paths)
        question_context = {
            "worker_task": agent_task,
            "inputs": inputs_dict,
            "proposals": paths_str,
        }
        statep.publish(question_context, self.version_id, self.is_static)
        self.version_id += 1
        self.status = ModuleStatus.SUCCESS


class UniversalSelfConsistency(EnsembleOptionBase):
    aggregator_system_prompt = """
You are tasked with aggregating multiple proposals generated by a worker agent in response to a specific task. Your job is to analyze the proposals, identify the points of agreement, and craft a final answer that reflects the majority consensus and maintains coherence. Where proposals conflict, your goal is to resolve discrepancies by selecting the most consistent or widely agreed-upon points. Ensure that the final answer is comprehensive, accurate, and respects the worker’s role and expertise.

You will be provided with:
- The role of that worker agent, since it is a LLM-agent so it will be its system prompt.
- Input to the worker agent, upon which the proposals were generated.
- A set of responses provided by the worker agent, each with potentially agreed or differing content. Each worker responses includes its reasoning and a final answer. 

Please read through all the responses carefully and provide a clear, consistent answer that respects the worker’s expertise and integrates the most consistent and widely agreed-upon content from the multiple proposals. 
"""

    def __init__(
        self,
        num_path: int,
        temperature: float = 0.7,
        change_temperature: bool = True,
    ):
        super().__init__("universal_self_consistency", num_path)
        self.temperature = temperature
        self.change_temperature = change_temperature

    def describe(self):
        temp_desc = (
            f"Temperature: {self.temperature}"
            if self.change_temperature
            else "No temperature change"
        )
        desc = f"""
        - Universal Self-Consistency Ensemble -
        Spawn <{self.num_path}> samplers ({temp_desc}) and aggregate the results. Aggregator is LLM-based and answers the question based on the majority consensus.
        """
        return desc

    def sample_then_aggregate(self, lm: Model) -> Module:
        sub_graph = Workflow(f"{lm.name}_ensemble_{self.name}")
        input_name, output_name = (
            f"{lm.name}_sub_graph_input",
            f"{lm.name}_sub_graph_output",
        )
        sub_graph.add_module(InputModule(input_name))
        sub_graph.add_module(Output(output_name))

        # Sampler
        lm_copies = [copy.deepcopy(lm) for _ in range(self.num_path)]
        for i, lm_copy in enumerate(lm_copies):
            lm_copy.name = f"{lm.name}_sampler_{i}"
            lm_copy.reset()
            sub_graph.add_module(lm_copy)
            sub_graph.add_edge(input_name, lm_copy.name)

        sampler_post_process = SamplerPostProcess(
            name=f"{lm.name}_sampler_post_process",
            origin_expert=lm,
            experts=lm_copies,
        )
        sub_graph.add_module(sampler_post_process)
        sub_graph.add_edge(
            [lm_copies.name for lm_copies in lm_copies], sampler_post_process.name
        )

        custom_format_instruction = "Please only give the final answer"
        if lm.contains_custom_format_instructions():
            custom_format_instruction = f"\nAnswer format instructions given to the worker: {lm.get_custom_format_instructions_if_any()}\n Please strictly follow this as if you are generating the answer on worker's behalf."

        if isinstance(lm, StructuredModel):
            new_output_format = OutputFormat(
                lm.output_format.schema,
                lm.output_format.should_hint_format_in_prompt,
                custom_format_instruction,
            )
            agg_agent = StructuredModel(
                f"{lm.name}_aggregator",
                UniversalSelfConsistency.aggregator_system_prompt,
                input_variables=[
                    Input("worker_task"),
                    Input("inputs"),
                    Input("proposals"),
                ],
                output_format=new_output_format,
                lm_config=copy.deepcopy(lm.lm_config),
            )
        else:
            new_output_label = OutputLabel(
                lm.get_output_label_name(), custom_format_instruction
            )
            agg_agent = Model(
                f"{lm.name}_aggregator",
                UniversalSelfConsistency.aggregator_system_prompt,
                input_variables=[
                    Input("worker_task"),
                    Input("inputs"),
                    Input("proposals"),
                ],
                output=new_output_label,
                lm_config=copy.deepcopy(lm.lm_config),
            )
        sub_graph.add_module(agg_agent)
        sub_graph.add_edge(sampler_post_process.name, agg_agent.name)

        sub_graph.compile()
        return sub_graph

    @classmethod
    def from_dict(cls, meta):
        num_path = meta["num_path"]
        temperature = meta["temperature"]
        change_temperature = meta.get("change_temperature")
        return cls(num_path, temperature, change_temperature)

    def to_dict(self):
        base = super().to_dict()
        base["num_path"] = self.num_path
        base["temperature"] = self.temperature
        base["change_temperature"] = self.change_temperature
        return base
