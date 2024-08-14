from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def format_qa_pairs(questions, answers):
    """Format Q and A pairs"""
    
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

def answer_compose_kernel(question, sub_questions, sub_answers):
    context = format_qa_pairs(sub_questions, sub_answers)

    # Prompt
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = answer_compose_kernel.lm
    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    final_answer = final_rag_chain.invoke({"context": context, "question": question})
    return {'answer': final_answer}