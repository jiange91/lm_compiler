from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

### Answer Grader
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

def answer_grader_kernel(question, generation):
    llm = answer_grader_kernel.lm
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    answer_grader = answer_prompt | structured_llm_grader
    score = answer_grader.invoke({"question": question, "generation": generation}).binary_score
    return {'answer_score': score}
    