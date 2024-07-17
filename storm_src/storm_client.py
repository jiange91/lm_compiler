import dspy
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import os

from persona_generator import CreateWriterWithPersona
from compiler.IR.program import Workflow, Module, StatePool
from compiler.dspy_bridge.interface import DSPyLM
from compiler.optimizer.bootstrap import BootStrapLMSelection
from common import StormState

@dataclass
class STORMWikiRunnerArguments:
    """Arguments for controlling the STORM Wiki pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."},
    )
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of questions in conversational question asking."},
    )
    max_perspective: int = field(
        default=5,
        metadata={"help": "Maximum number of perspectives to consider in perspective-guided question asking."},
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of search queries to consider in each turn."},
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, disable perspective-guided question asking."},
    )
    search_top_k: int = field(
        default=3,
        metadata={"help": "Top k search results to consider for each search query."},
    )
    retrieve_top_k: int = field(
        default=3,
        metadata={"help": "Top k collected references for each section title."},
    )
    max_thread_num: int = field(
        default=10,
        metadata={"help": "Maximum number of threads to use. "
                          "Consider reducing it if keep getting 'Exceed rate limit' error when calling LM API."},
    )


# --------------------------------------------
# Compiler Client
# --------------------------------------------

# Initialize State
state = StormState()
state.publish({'topic': 'Taylor Hawkins',
               'max_num_persona': 5})

# Get persona writer module
get_personas = CreateWriterWithPersona()
find_related_topic_module = DSPyLM('find_related_topics', get_personas.agent_get_topics)
get_personas_module = DSPyLM('get_personas', get_personas.agent_get_personas)

storm_workflow = Workflow()
storm_workflow.add_module(find_related_topic_module)
storm_workflow.add_module(get_personas_module)
storm_workflow.add_edge(find_related_topic_module, get_personas_module)
storm_workflow.set_root(find_related_topic_module)

openai_kwargs = {
    'api_key': os.getenv("OPENAI_API_KEY"),
    'api_provider': os.getenv('OPENAI_API_TYPE'),
    'temperature': 1.0,
    'top_p': 0.9,
}

# Sample run
# find_related_topic_module.lm_config = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, **openai_kwargs}
# get_personas_module.lm_config = {'model': 'gpt-3.5-turbo', 'max_tokens': 500, **openai_kwargs}
# storm_workflow.run(state=state)
# print(state.state)

# Bootstrap
find_related_topic_module.lm_config = {'max_tokens': 500, **openai_kwargs}
get_personas_module.lm_config = {'max_tokens': 500, **openai_kwargs}

lm_options = ['gpt-3.5-turbo', 'gpt-4o']
def lm_metric(x):
    return 0.5

ms_boot = BootStrapLMSelection(
    workflow=storm_workflow,
    teachers='gpt-3.5-turbo',
    module_2_options=lm_options,
    module_2_metric=lm_metric,
)

ms_boot.bootstrap(trainset=[state])
