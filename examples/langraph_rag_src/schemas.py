from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from pydantic import BaseModel, Field
from typing import Literal

class HypertheticalPassages(BaseModel):
    """return the topics/passages generated by HyDE"""
    
    sub_questions : list[str] = Field(
        description="a list of sub-topics"
    )
    passages: list[str] = Field(
        description="passages generated for the sub-topics"
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class KnowledgeAnswer(BaseModel):
    """Knowledgable answer based on retrieved documents for the question"""
    knowledge: str = Field(
        description="answer for question"
    )

class FinalAnswer(BaseModel):
    """return the answer generated by RAG"""
    answer: str = Field(
        description="answer for the main question"
    )

class VerifyDecision(BaseModel):
    """return the decision of KALMV"""

    decision: Literal["ae", "re", "ge", "accept"] = Field(
        description="Given user question, knowledge, answer, determine if the knowledge or answer is deficient.",
    )