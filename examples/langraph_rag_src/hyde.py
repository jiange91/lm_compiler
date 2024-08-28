from langchain.prompts import ChatPromptTemplate, PromptTemplate

# HyDE document genration
template = """Please write a paragraph to answer the question
Question: {question}
Paragraph:"""
prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

def hyde_kernel(sub_questions):
    llm = hyde_kernel.lm
    hyde = prompt_hyde | llm | StrOutputParser()
    inputs = [{"question": question} for question in sub_questions]
    ps = hyde.batch(inputs)
    # ps = []
    # for question in sub_questions:
    #     ps.append(hyde.invoke({"question": question}))
    return {'passages': ps}
