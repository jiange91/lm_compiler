
complexity_system = """
You are an expert at designing LLM-based agent workflow. Your task is to evaluate the complexity of the responsibility of each agent. 

You will be provided with their system prompts for reference. Please assign each agent a numerical rating on the following scale to indicate its complexity. 

You should consider:
Does the task encompass a wide range of horinzontal responsibilities?
Does the task require multi-step reasoning, planning, decision-making? 
The difficulty of each sub-tasks that involved in the subject.

For each agent, please give your rate from 1 to 5 along with your rationale for the rating.
Rating criteria: 
1: straightforward, the task is simple and clear.
2: given problem has multiple horizontal sub-tasks, but all are simple to tackle.
3: given problem requires vertical planning, reasoning, decision-making, but all sub-tasks are very simple.
4: At least one sub-task is non-trivial to solve, can benefit from task decomposition for better interpretability and ease of reasoning.
5: Task is ambiguous, complex, and requires very fine-grained decomposition.

Your answer should follow the order of the given agents.
"""

decompose_system = """
You are an expert at designing LLM-based agent workflow. Your task is to design a set of agents to perform the given task, each with a clear and separate role. 

The given task is originally performed by a single LLM agent. You will be provided with its prompt for task description. Please pay attention to all the details in the prompt. Your should make sure that the new agent system should fulfill all the requirements in the original prompt. Also you will be provided with some rationale for the complexity of the task to help you decompose the task.

Principles:
1. In all new prompts, please be clear about the role of the agent and provide detailed instruction to guide the agent to perform its task. Do not include information of how to interact with other agents in the prompt, e.g. "pass the output to the next agent", focus on the task of the agent itself.
2. Decomposed agents should be independent and have clear responsibilities.
3. For each new agent, you should only use the information provided in the original prompt or the information generated by other new agents. Do not seek information from external sources like web-search or databases that do not exist in the orignal prompt.

For the final output, you need to specify the prompt for each agent in your refined workflow and give each agent a concise and descriptive name.

Avoid adding trivial LLM-agents who do not contribute much to the overall task or can be easily merged with other agents or be replaced by a simple rule-based functions.

As an example, given the original single-agent system prompt:

'''
You are an expert at assessing the quality of a software given user requirements. You should evaluate t based on the following criteria:
1. If software crashes, grade it as 'fail'.
2. otherwise, if the software does not meet requirements, grade it as 'wrong'.
3. otherwise, if the software is slow, grade it as 'slow'.
4. otherwise, grade it as 'accept'.
'''

You can decompose this system into three agents: testing agent, product manager agent, and performance agent. Each agent has a more focused role than the original system. But you should not add an agent specifically for the 'accept' case, as it can be easily handled by if-else condition after the performance agent.
"""

example_agent_info = """
{
    "agent_prompt": "\nYou are an expert at assessing the quality of a software given user requirements. You should evaluate t based on the following criteria:\n1. If software crashes, grade it as 'fail'.\n2. otherwise, if the software does not meet requirements, grade it as 'wrong'.\n3. otherwise, if the software is slow, grade it as 'slow'.\n4. otherwise, grade it as 'accept'.\n",
    "input_names": [
        "requirement",
        "software"
    ],
    "output_names": [
        "decision"
    ]
}
"""

decompose_refine_system = """
You are an expert at designing LLM multi-agent workflow. Your task is to rewrite a single-LLM-agent system following some given guidance. 

You will be provided with information about the existing agent system, including its input/output variable name, and the high-level prompt. The guidance is a group of suggested agents you should use in the new system. Each includes a name and a prompt. They should collaboratively achieve the same goal as the original system. You need to decide how suggested new agents interact with each other. 

Specifically, for each new agent:
1. Decide the input/output variable name of each new agent.
2. Decide the set of agents that will "potentially" take action next. Use 'END' as the agent name to indicate no more agents to invoke along this path. Please list all possible agents that can be invoked next.
3. If next agents to be invoked need to decide dynamically, please write a python method for deciding this at the runtime. This is important as invoking wrong/unrelated agents can lead to incorrect results and waste compute budget. Otherwise you can put a "None" for this field.
4. Enrich the proposed prompt if necessary so that the agent can learn to take advantage of its input and also provide the expected output.

In your answer, please include your decision for all agents in the new system.

Be careful with the input/output variable names of new agents. 
 - Keep in mind that the new system will replace the existing one, so you should only use input keywords that present in the original agent system or variables generated by agents in the new system. 
 - Also make sure outputs of the original system will always be generated by the new agent system no matter what control flow is taken. In other words, any agent whose next action may be 'END' should generate at least part of the original output varaible.
 - Do not generate unnecessary outputs, be more focused.

As an example:

## Information of the existing agent system
{example_agent_info}

## Suggested new agents, with name and prompt
{{
    "TestingAgent": "You are an expert at testing softwares given the code.", 
    "ProductManagerAgent": "You are an expert at evaluating if the product meets user requirement",
    "PerformanceAgent": "You are an expert at evaluating the performance of the software",
}}

Peudo Output in this case, this is an example to help you learn the general good design:

{example_json_output}

In this example you can see that each agent have a more focused role than the original system. It decompose the original multi-step reasoning process: testing -> requirement evaluation -> performance evaluation, into separate agents with different specializations.

Also dependencies between new agents are well-designed so that the new system will always generate the correct output. If the software crashes, the system will output 'fail' for variable 'decision'. If the software does not meet requirements, the system will output 'wrong' for variable 'decision'. If the software is slow, the system will output 'slow' for variable 'decision'. If survived all these checks, the system will output 'accept' for variable 'decision'. So it matches the original system's output.

"""


refine_example_json_output = """
{
  "agents": {
    "TestingAgent": {
      "inputs": ["code", "requirements"],
      "outputs": ["decision"],
      "prompt": "You are an expert in software testing, specializing in generating test cases and executing them based on the provided code. Your primary objective is to ensure that the software is reliable, functional, and free from defects by creating comprehensive test scenarios based on user requirements, and analyzing the results. Rate the software as 'fail' if it crashes during any test cases, otherwise 'test_pass'",
      "next_action": ["ProductManagerAgent", "END],
      "dynamic_action_decision": "def next_agent(decision): return ['END'] if decision == 'fail' else ['ProductManagerAgent']"
    },
    "ProductManagerAgent": {
      "inputs": ["requirements", "code"],
      "outputs": ["decision"],
      "prompt": "You are an expert in evaluating software products to determine if they meet user requirements. Your primary objective is to thoroughly assess the product's features, functionality, and performance to ensure that it aligns with the specified user needs and expectations. If the software does not meet the requirements, output 'wrong'; otherwise, output 'requirement_fulfilled'.",
      "next_action": ["PerformanceAgent", "END"],
      "dynamic_action_decision": "def next_agent(decision): return ['END'] if decision == 'wrong' else ['PerformanceAgent']"
    },
    "PerformanceAgent": {
      "inputs": ["code"],
      "outputs": ["decision"],
      "prompt": "You are an expert at reviewing code. If you find any potential performance issues in the code, output 'slow'; otherwise, output 'accept'",
      "next_action": ["END"],
      "dynamic_action_decision": "None"
    }
  }
}
"""

mid_level_system_format_instructions = f'''
Specifically, you need to make sure the output JSON schema can be used to initialize the pydantic model "NewAgentSystem" defined as follows:

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
        'python code for dynamically deciding the next action, put "None" if not needed'
    )
    
class NewAgentSystem(BaseModel):
    """New agent system"""
    agents: Dict[str, AgentMeta] = Field(
        description="dictionary of agent name to information about that agent"
    )
  
Example output:
{refine_example_json_output}
'''


finalize_new_agents_system = """
You are an expert at designing LLM multi-agent workflow. Your team just re-wrote a single-agent system into a multi-agent system but haven't finalized it yet. Now, you need to align the new system to the functionality of the original system.

You will be provided with both information for the old single-agent system and the new multi-agent system.

You should check if the output of old-system can be generated by the new system no matter what code path is taken. If not, make necessary changes to the output varaible of new agents. When you make changes, make sure the input variable name in 'dynamic_action_decision' field is also updated to reflect the new output varaible name.

Also you should decide the output json schema for each agent in the new system. If agents are providing part of the final output (maybe in a different schema format), you need to aggregate them to align with the single-agent systen's output schema, you should decide how to combine them using python code. If you think no aggregation logic is needed, just put 'None' for this. Make sure that at the end of the execution of the new system, the final output is compatible with the original output schema.

Here's an example:

## Information of the old single-agent system
{
    "agent_prompt": "\nYou are a grader assessing relevance of a retrieved document to a user question.\nIf the document either 1. does not contains semantic meaning related to the user question, or 2. contains toxic content.\n You should give it a binary score 'no' to reject the document, otherwise 'yes'.",
    "input_varaibles": [
        "sub_question",
        "doc_for_filter"
    ],
    "output_json_schema": {
        "title": "GradeDocuments",
        "description": "Binary score for relevance check on retrieved documents.",
        "type": "object",
        "properties": {
            "binary_score": {
                "title": "Binary Score",
                "description": "Documents are relevant to the question, 'yes' or 'no'",
                "type": "string"
            }
        },
        "required": [
            "binary_score"
        ]
    }
}

You can see this agent takes as input a sub-question and a document to filter. It gives a single binary score as output. The variable name for this agent's output is "binary_score". So you need to make sure the new system should also always generate this output variable in all code paths.

## Information of the suggested multi-agent system 
{
    "agents": {
        "RelevanceEvaluationAgent": {
            "inputs": [
                "sub_question",
                "doc_for_filter"
            ],
            "outputs": [
                "relevance_score"
            ],
            "prompt": "Your role is to evaluate the relevance of the provided document to the user question. If the knowledge is irrelevant to the question, grade it as 'No'. If the knowledge is relevant, pass the evaluation to the next agent.",
            "next_action": ["END", "ToxicModerationAgent"],
            "dynamic_action_decision": "def next_agent(relevance_score): return ['END'] if relevance_score == 'no' else ['ToxicModerationAgent']"
        },
        "ToxicModerationAgent": {
            "inputs": [
                "doc_for_filter"
            ],
            "outputs": [
                "binary_score"
            ],
            "prompt": "You are responsible for assessing whether the provided document contains toxic content. Answer with 'yes' if the document is toxic, otherwise 'no'.",
            "next_action": ["END"],
            "dynamic_action_decision": "None"
        }
    }
}

Let me explain the information format for the new multi-agent system first. Overall it presents a dictionary of agent names to information about that agent.

Each agent has a list of inputs/output varaible names, a prompt, and a "next_action" field. The next_action field can be a list of all possible agents to invoke next. The dynamic_action_decision field is a python code that decides the next agent to invoke at runtime. If it's "None", it means all agents listed in "next_action" field will be invoked deterministically.

Specifically for "next_action" field,
'END' is a special keyword to indicate that: along this path, there will be no more agents to invoke. This does not mean the system will end immediately, the system will end when there are no more agents in execution.

## Solution
As you can see this new design has a big problem, the relevance evaluation agent will dynamically decide the next agent to invoke. If it decides to end, the output "binary_score" required by the old system, will not be generated since the current output variable name for this agent is "relevance_score". 

Because this output has the same semantic meaning of the required output, your fix is simple - change the output variable name of the relevance evaluation agent to "binary_score".

so final aggregator function is "None" in this case. Also dont forget to update the 'dynamic_action_decision' field accordingly as this:
"dynamic_action_decision": "def next_agent(binary_score): return ['END'] if binary_score == 'no' else ['ToxicModerationAgent']"

Here's the example output schema for each agent in the new system after fix:

For Relevance Evaluation Agent:
{
    "title": "RelevanceScoreSchema",
    "description": "Binary score for relevance check on retrieved documents.",
    "type": "object",
    "properties": {
        "binary_score": {
            "title": "Binary Score",
            "description": "Documents are relevant to the question, 'yes' or 'no'",
            "type": "string"
        }
    },
    "required": [
        "binary_score"
    ]
}

For Toxic Moderation Agent:
{
    "title": "ToxicScoreSchema",
    "description": "Binary score for toxicity check on retrieved documents.",
    "type": "object",
    "properties": {
        "binary_score": {
            "title": "Binary Score",
            "description": "Documents are toxic, 'yes' or 'no'",
            "type": "string"
        }
    },
    "required": [
        "binary_score"
    ]
}

You should generate valid json schema for each agent. Their output variables should be included as top-level properties in the schema. You are free to use any descriptive title and description for the schema.

Let me give you another example that require more complex changes:

## Information of the old single-agent system
{
    "agent_prompt": "You are an expert at reviewing research papers. You are tasked with evaluating the presentation and novelty of each paper.",
    "input_varaibles": [
        "papers"
    ],
    "output_json_schema": {
        "title": "PaperReviews",
        "description": "Reviews a list of research papers",
        "type": "object",
        "properties": {
            "scores": {
                "title": "Scores",
                "description": "dictionary of paper title to its score",
                "type": "object",
                "additionalProperties": {
                    "$ref": "#/definitions/Score"
                }
            }
        },
        "required": [
            "scores"
        ],
        "definitions": {
            "Score": {
                "title": "Score",
                "description": "Ratings of a research paper",
                "type": "object",
                "properties": {
                    "presentation": {
                        "title": "Presentation",
                        "description": "Presentation of the research paper",
                        "type": "integer"
                    },
                    "novelty": {
                        "title": "Novelty",
                        "description": "Novelty of the research paper",
                        "type": "integer"
                    }
                },
                "required": [
                    "presentation",
                    "novelty"
                ]
            }
        }
    }
}

In this example, the old system takes a list of papers as input and outputs a dictionary of paper title to its score. Each score has two fields: presentation and novelty.

## Information of the suggested multi-agent system
{
    "agents": {
        "PresentationEvaluationAgent": {
            "inputs": [
                "papers"
            ],
            "outputs": [
                "presentation_scores"
            ],
            "prompt": "Your role is to evaluate the presentation of each paper. Output a dictionary of paper title to its presentation score.",
            "next_action": ["END"],
            "dynamic_action_decision": "None"
        },
        "NoveltyEvaluationAgent": {
            "inputs": [
                "papers"
            ],
            "outputs": [
                "novelty_scores"
            ],
            "prompt": "Your role is to evaluate the novelty of each paper. Output a dictionary of paper title to its novelty score.",
            "next_action": ["END"],
            "dynamic_action_decision": "None"
        }
    }
}

## Solution
In this case both agent gives their part of outputs and you can't simply rename these variables to the original output keyword because they are semantially different. Instead, you can keep these outputs and write a aggregator function to synthesis the final output. 

You can define the json schema of each agent's output as follows:

For Presentation Evaluation Agent:
{
    "title": "PresentationScoresSchema",
    "description": "Scores for presentation of research papers",
    "type": "object",
    "properties": {
        "presentation_scores": {
            "title": "Presentation Scores",
            "description": "dictionary of paper title to its score",
            "type": "object",
            "additionalProperties": {
                "type": "integer"
            }
        }
    },
    "required": [
        "presentation_scores"
    ]
}

For Novelty Evaluation Agent:
{
    "title": "NoveltyScoresSchema",
    "description": "Scores for novelty of research papers",
    "type": "object",
    "properties": {
        "novelty_scores": {
            "title": "Novelty Scores",
            "description": "dictionary of paper title to its score",
            "type": "object",
            "additionalProperties": {
                "type": "integer"
            }
        }
    },
    "required": [
        "novelty_scores"
    ]
}

Then, you need to write a function to combine the outputs of these new agents to generate the final output that is compatible to the orignal final output schema "PaperReviews". You should use their variable name in the function signature as they will be passed as keyword arguments in the future call. The first argument of the aggregation function should always be named "output_schema", which is a pydantic model class that represents the final output schema.

Also this function need to return a dictionary that is compatible with the final output schema. In this case, the schema is "PaperReviews". so the final dictionary will be something like this:
{
    "scores": {
        "paper1": {"presentation": 5, "novelty": 3},
        "paper2": {"presentation": 4, "novelty": 2}
}

final output aggregator example code:
```python
def combine_outputs(output_schema: PaperReviews, presentation_scores: dict[str, int], novelty_scores: dict[str, int]):
    scores = {}
    for title in presentation_scores:
        scores[title] = {"presentation": presentation_scores[title], "novelty": novelty_scores[title]}
    return output_schema(scores=scores)
```
please omit type hint when you generate the answer, this demonstration is for you to better understand the function signature.
"""

structured_system_format = '''
The generated output will be used to initialize the StructuredAgentSystem pydantic model, which is defined as follows:

class AgentSemantic(BaseModel):
    """Information about each agent"""
    agent_prompt: str = Field(
        description="prompt for the agent"
    )
    inputs_varaibles: list[str] = Field(
        description="list of input variables for the agent"
    )
    output_json_schema: str = Field(
        description="json output schema for the agent"
    )
    next_action: List[str] = Field(
        description="all possible next agents to invoke"
    )
    dynamic_action_decision: str = Field(
        "python code for dynamically deciding the next action, put 'None' if not needed"
    )
    
class StructuredAgentSystem(BaseModel):
    """Refined agent system with structured output schema"""
    agents: dict[str, AgentSemantic] = Field(
        description="dictionary of agent name to information about that agent"
    )
    
    final_output_aggregator_code: str = Field(
        description="python code to combine the outputs of the new agents to generate the final output, put 'None' if not needed"
    )

The generated json output should be compatible with the StructuredAgentSystem pydantic model, for example:
{
    "agents": ...
    "final_output_aggregator_code": ...
}
'''