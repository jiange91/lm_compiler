from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


#--------------------- Verifier ---------------------#
class VerifyDecision(BaseModel):
    """return the decision of KALMV"""

    decision: Literal["re", "ge", "correct"] = Field(
        description="Given user question, knowledge, answer, determine if the knowledge or answer is deficient.",
    )

# Prompt
system = """
You are an expert at evaluating knowledge augmented LLM generation. Given the user question, knowledge, and answer, make the following decisions: 
1. If the knowledge is irrelevant to the question, grade it as 're'.
2. otherwise, if the answer is not grounded in the knowledge, grade it as 'ge'.
3. otherwise, grade it as 'correct'.
"""

verify_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n{question} \n\n"
                  "Knowledge: \n\n{knowledge} \n\n"
                  "Answer: \n\n{answer} \n\n"),
    ]
)


def kalmv_kernel(question, answer, sub_answers):
    llm = kalmv_kernel.lm
    structured_llm_verifier = llm.with_structured_output(VerifyDecision)
    verifier = verify_prompt | structured_llm_verifier
    decision = verifier.invoke({"question": question, "knowledge": "\n\n".join(sub_answers), "answer": {answer}}).decision
    return {'decision': decision}
    