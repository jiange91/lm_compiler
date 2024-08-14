from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

### Question Re-writer

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

def query_rewriter_kernel(question):
    llm = query_rewriter_kernel.lm
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return {'question': question_rewriter.invoke({"question": question})} 