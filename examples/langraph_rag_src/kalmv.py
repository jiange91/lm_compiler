from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


#--------------------- Verifier ---------------------#
class VerifyDecision(BaseModel):
    """return the decision of KALMV"""

    decision: Literal["ae", "re", "ge", "accept"] = Field(
        description="Given user question, knowledge, answer, determine if the knowledge or answer is deficient.",
    )

# Prompt
system = """
You are an expert at evaluating knowledge augmented LLM generation. Given the user question, knowledge, and answer, make the decision in the following order:
1. If the answer does not address the question, grade it as 'ae'.
2. If the knowledge is irrelevant to the question, grade it as 're'.
2. If the answer is not grounded by the knowledge, grade it as 'ge'.
3. otherwise, grade it as 'accept'.

Only one decision should be made.
"""

verify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n{question} \n\n"
                  "Knowledge: \n\n{knowledge} \n\n"
                  "Answer: \n\n{answer} \n\n"),
    ]
)


def kalmv_kernel(llm, question, answer, sub_answers):
    structured_llm_verifier = llm.with_structured_output(VerifyDecision)
    verifier = verify_prompt | structured_llm_verifier
    decision = verifier.invoke({"question": question, "knowledge": "\n\n".join(sub_answers), "answer": {answer}}).decision
    return {'decision': decision}
    
#------------------------ New Semantic ------------------------#
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM

kalmv_semantic = LangChainSemantic(
    system_prompt=system,
    inputs=['question', 'answer', 'knowledge_answer'],
    output_format=VerifyDecision,
)

kalmv_module = LangChainLM(
    name="kalmv",
    semantic=kalmv_semantic,
)