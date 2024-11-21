import dotenv
import cognify

dotenv.load_dotenv()

# Define system prompt
system_prompt = """
You are an expert at answering questions based on provided documents. Your task is to provide the answer along with all supporting facts in given documents.
"""

# Initialize the model
import cognify
lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {
        'temperature': 0.0,
    }
)

# Define agent routine 

cognify_qa_agent = cognify.Model(
    agent_name="qa_agent",
    system_prompt=system_prompt,
    input_variables=[
        cognify.Input(name="question"), 
        cognify.Input(name="documents")
    ],
    output=cognify.OutputLabel(name="answer"),
    lm_config=lm_config
)

# Use builtin connector for smooth integration
qa_agent = cognify.as_runnable(cognify_qa_agent) 

from cognify.optimizer import register_opt_workflow

def doc_str(docs):
    context = []
    for i, c in enumerate(docs):
        context.append(f"[{i+1}]: {c}")
    return "\n".join(docs)

@register_opt_workflow
def qa_workflow(question, documents):
    format_doc = doc_str(documents)
    answer = qa_agent.invoke({"question": question, "documents": format_doc}).content
    return {'answer': answer}    
