from langchain import hub
from langchain_core.output_parsers import StrOutputParser

from concurrent.futures import ThreadPoolExecutor

# Prompt
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = hub.pull("rlm/rag-prompt")


def generator_kernel(sub_questions, documents):
    llm = generator_kernel.lm
    rag_chain = prompt | llm | StrOutputParser()
    inputs = [{"context": document, "question": question} for question, document in zip(sub_questions, documents)]
    sub_answers = rag_chain.batch(inputs)
    # for question, document in zip(sub_questions, documents):
    #     sub_answers.append(rag_chain.invoke({"context": document, "question": question}))
    return {'sub_answers': sub_answers}