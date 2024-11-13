import dotenv
from cognify._logging import _configure_logger
import copy

dotenv.load_dotenv()
_configure_logger("INFO")

from pydantic import BaseModel
from typing import List

# Define system prompt
system_prompt = """
You are an expert at answering questions based on provided documents. 
"""

# Initialize the model
from cognify.llm.model import LMConfig
lm_config = LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

# Define agent routine 
from cognify.llm.model import CogLM, InputVar, OutputLabel
cognify_qa_agent = CogLM(
    agent_name="qa_agent",
    system_prompt=system_prompt,
    input_variables=[InputVar(name="question"), InputVar(name="documents")],
    output=OutputLabel(name="response"),
    lm_config=lm_config,
)

def doc_str(docs):
    context = []
    for i, c in enumerate(docs):
        context.append(f"[{i+1}]: {c}")
    return "\n".join(docs)


from langgraph.graph import END, START, StateGraph, MessagesState
from typing import Dict, TypedDict

class State(TypedDict):
    question: str
    documents: List[str]
    answer: str

from cognify.optimizer.plugin import Entry
class HelloWorld(Entry):
    def __init__(self):
        self.cognify_agent = cognify_qa_agent
        self.app = self.build_workflow()
        
        from cognify.frontends.langchain.connector import as_runnable
        self.qa_agent = as_runnable(self.cognify_agent)
    
    def build_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("grounded_qa", self.qa_agent_routine)
        workflow.add_edge(START, "grounded_qa")
        workflow.add_edge("grounded_qa", END)

        app = workflow.compile()
        return app

    def qa_agent_routine(self, state):
        question = state["question"]
        documents = state["documents"]
        format_context = doc_str(documents)
        result = {"answer": self.qa_agent.invoke({"question": question, "documents": format_context}).content}
        print(result)
        return result
    
    def forward(self, input):
        response = self.app.invoke(
            {"question": input[0], "documents": input[1]}
        )
        return response['answer']

if __name__ == "__main__":
    input = {
        "question": "What was the 2010 population of the birthplace of Gerard Piel?", 
        "documents": [
            'Gerard Piel | Gerard Piel (1 March 1915 in Woodmere, N.Y. – 5 September 2004) was the publisher of the new Scientific American magazine starting in 1948. He wrote for magazines, including "The Nation", and published books on science for the general public. In 1990, Piel was presented with the "In Praise of Reason" award by the Committee for Skeptical Inquiry (CSICOP).',
            'Woodmere, New York | Woodmere is a hamlet and census-designated place (CDP) in Nassau County, New York, United States. The population was 17,121 at the 2010 census.',
        ],
    }

    entry = HelloWorld()
    
    new_cognify_qa_agent = CogLM(
        agent_name="new_qa_agent",
        system_prompt="You are an expert at answering questions based on provided documents. Please be very concise.",
        input_variables=[InputVar(name="question"), InputVar(name="documents")],
        output=OutputLabel(name="response"),
        lm_config=lm_config,
    )
    
    new_entry = copy.deepcopy(entry)
    new_entry.cognify_agent.invoke = new_cognify_qa_agent.invoke
    new_entry.app = new_entry.build_workflow()
    
    # entry.cognify_agent.invoke = new_cognify_qa_agent.invoke
    
    result = new_entry.forward((
        input["question"],
        input["documents"]
    ))
    print(result)