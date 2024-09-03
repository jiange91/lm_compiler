from langchain_core.pydantic_v1 import BaseModel, Field
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.program import Workflow, hint_possible_destinations, Context
from compiler.IR.modules import Map, CodeBox

# ------------------------ Original IR ------------------------#
workflow = Workflow('rag')

old_prompt = """
You are a grader assessing a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: list[str] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

old_llm_semantic = LangChainSemantic(
    system_prompt=old_prompt,
    inputs=['question', 'documents'],
    output_format=GradeDocuments,
)

old_agent = LangChainLM('doc_relevance', old_llm_semantic)
workflow.add_module(old_agent)

# ------------------------ After decomposition ------------------------#
word_match_prompt = """
You are an expert in keyword matching, focusing on determining whether a document contains specific keywords related to a user’s question. Your primary objective is to identify and confirm the presence of these keywords in the document.
"""

class KeyWordMatch(BaseModel):
    """Binary score for keyword matching of retrieved documents."""

    km_score: list[str] = Field(
        description="Documents contain keyword related to the question, 'yes' or 'no'"
    )

agent_1_semantic = LangChainSemantic(
    system_prompt=word_match_prompt,
    inputs=['question', 'documents'],
    output_format=KeyWordMatch,
)

semantic_analysis_prompt = """
You are an expert in semantic analysis, specializing in understanding and evaluating the meaning and context of content in relation to a user’s question. Your primary objective is to determine if the document’s content is semantically relevant to the user’s question, even if the exact keywords are not present.
"""

class SemanticAnalysis(BaseModel):
    """Binary score for semantic analysis of retrieved documents."""

    sa_score: list[str] = Field(
        description="Documents are semantically relevant to the question, 'yes' or 'no'"
    )

agent_2_semantic = LangChainSemantic(
    system_prompt=semantic_analysis_prompt,
    inputs=['question', 'documents'],
    output_format=SemanticAnalysis,
)

def procuce_old_output_schema(km_score, sa_score):
    def criteria(km, sa):
        if km == 'yes' and sa == 'yes':
            return 'yes'
        return 'no'
    return GradeDocuments(binary_score=map(criteria, km_score, sa_score))

new_system_post_process = CodeBox(
    name="doc_relevance_post_process",
    kernel=procuce_old_output_schema,
)
retain_old_output = CodeBox(
    name="retain_old_output",
    kernel=new_system_post_process
)

new_sub_graph_to_replace_old_agent = Workflow('new_sub_graph_doc_relevance')
new_sub_graph_to_replace_old_agent.add_module(LangChainLM('keyword_match', agent_1_semantic))
new_sub_graph_to_replace_old_agent.add_module(LangChainLM('semantic_analysis', agent_2_semantic))
new_sub_graph_to_replace_old_agent.add_edge(new_sub_graph_to_replace_old_agent.start, 'keyword_match')
new_sub_graph_to_replace_old_agent.add_edge(new_sub_graph_to_replace_old_agent.start, 'semantic_analysis')
new_sub_graph_to_replace_old_agent.add_edge(['keyword_match', 'semantic_analysis'], new_system_post_process)

# Replace the old agent with the new sub-graph
workflow.replace_node(old_agent, new_sub_graph_to_replace_old_agent)