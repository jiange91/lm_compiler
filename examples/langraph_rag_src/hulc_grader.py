from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

### Hallucination Grader
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# LLM with function call
def hulc_grader_kernel(answer, sub_answers):
    llm = hulc_grader_kernel.lm
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_grader = hallucination_prompt | structured_llm_grader
    score = hallucination_grader.invoke({"documents": sub_answers, "generation": answer}).binary_score
    return {'hallucination_score': score}