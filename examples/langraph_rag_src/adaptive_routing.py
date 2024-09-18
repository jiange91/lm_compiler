from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


#--------------------- Router ---------------------#
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

def router_kernel(sub_questions):
    llm = router_kernel.lm
    structured_llm_router = llm.with_structured_output(RouteQuery)
    question_router = route_prompt | structured_llm_router
    data_sources = []
    for question in sub_questions:
        data_sources.append(question_router.invoke({"question": question}).datasource)
    return {'datasources': data_sources}
    