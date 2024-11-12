import dotenv

# Load the environment variables
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List


# Define system prompt
system_prompt = """
You are an expert at answering questions based on provided documents. Your task is to provide the answer along with all supporting facts in given documents.
"""

# Define Pydantic model for structured output
class AnswerOutput(BaseModel):
    answer: str
    supporting_facts: List[str]
    
# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(AnswerOutput)

# Define agent routine 
from langchain_core.prompts import ChatPromptTemplate
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "User question: {question} \n\nDocuments: {documents}"),
    ]
)

qa_agent = agent_prompt | model

# Define workflow

def doc_str(docs):
    context = []
    for i, c in enumerate(docs):
        context.append(f"[{i+1}]: {c}")
    return "\n".join(docs)

def qa_agent_routine(state):
    question = state["question"]
    documents = state["documents"]
    format_context = doc_str(documents)
    return {"response": qa_agent.invoke({"question": question, "documents": format_context})}

from langgraph.graph import END, START, StateGraph, MessagesState
from typing import Dict, TypedDict

class State(TypedDict):
    question: str
    documents: List[str]
    response: AnswerOutput
    
workflow = StateGraph(State)
workflow.add_node("grounded_qa", qa_agent_routine)
workflow.add_edge(START, "grounded_qa")
workflow.add_edge("grounded_qa", END)

app = workflow.compile()

input = {
    "question": "What was the 2010 population of the birthplace of Gerard Piel?", 
    "documents": [
        'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
        'Piet Ikelaar | Petrus "Piet" Gerardus Ikelaar (born 2 January 1896 Nieuwer Amstel, died Zaandam 25 November 1992) was a track cyclist from the Netherlands. He represented the Netherlands at the 1920 Summer Olympics. At his first appearance he won bronze medals in the 50 km track race and the 2000m tandem competition, alongside Frans de Vreng.',
        'Woodmere, New York | Woodmere is a hamlet and census-designated place (CDP) in Nassau County, New York, United States. The population was 17,121 at the 2010 census.',
        'Woodmere, Ohio | Woodmere is a village and eastern suburb of the Greater Cleveland area in the US state of Ohio. As of the 2010 census, Woodmere had a population of 884 residents. The village is bounded on the north by the city of Pepper Pike, on the west by the city of Beachwood, and on the south and east by the village of Orange.',
    ],
}

result = app.invoke(input)
print(result['response'])