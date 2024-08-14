from typing import Literal
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import numpy as np

subqs_format = NumberedListOutputParser()

#--------------------- decomposer ---------------------#
system = f"""
You are an expert at identifying facts and arguments in answering the question. Specifically, given a question answer pair, extract all statements in the given answer. Please identify all statements regardless its relevance to the question and its faithfulness.
{subqs_format.get_format_instructions()}
"""

statements_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\nAnswer: {answer}\nStatements: \n"),
    ]
)


def decompose_kernel(question, answer):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    statement_extractor = statements_prompt | llm | subqs_format
    statements = statement_extractor.invoke({"question": question, "answer": answer})
    return "\n".join(f'{i+1}. {s}' for i, s in enumerate(statements))

system_2 = f"""
You are an expert at analyzing whether the given statement is grounded by the factual context. Consider the given facts and following statements, then determine whether they
are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). 

Finally provide a final verdict for each statement at the end. Please respect the order of the statements and only say (Yes/No) at each item.
{subqs_format.get_format_instructions()}
"""

faith_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_2),
        ("human", "Context: {context}\n\nStatements: {Statements}\n"),
    ]
)

def check_faithfulness(statements, knowledges):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    faith_eval = faith_prompt | llm | subqs_format
    context_str = "\n".join(knowledges)
    faith_list = faith_eval.invoke({"context": context_str, "Statements": statements})
    # faithfulness is the 
    score = len([faith for faith in faith_list if faith.strip() == "Yes"]) / len(faith_list)
    return score

system = f"""
You are an expert at brainstorming the potential questions that can be derived from the given answer. Keep your questions open-ended and avoid yes/no questions. Each question should be one sentence long. Please only provide up to 5 questions.
{subqs_format.get_format_instructions()}
"""

qs_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Answer: {answer}\nQuestions: \n"),
    ]
)

def brainstorm_questions(answer):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    question_brainstorm = qs_prompt | llm | subqs_format
    questions = question_brainstorm.invoke({"answer": answer})
    return questions

def check_relevance(gold_q, reverse_qs):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    gold_vec = encoder.encode(gold_q)
    reverse_vecs = encoder.encode(reverse_qs)
    sims = cosine_similarity([gold_vec], reverse_vecs)[0]
    # estimate relevance 
    # to fuse with accuracy, rescale the relevance score to [0, 1], assume cosine similarity is in [-1, 1]
    relevance = np.mean((sims + 1) / 2)
    return relevance

def evaluate_rag_answer(question, answer, knowledges):
    statements = decompose_kernel(question, answer)
    faithfulness = check_faithfulness(statements, knowledges)
    questions = brainstorm_questions(answer)
    relevance = check_relevance(question, questions)
    
    # use harmonic mean to combine the scores
    score = 3 * (faithfulness * relevance) / (2 * faithfulness + relevance)
    return score

def evaluate_rag_answer_compatible(gold, pred, state):
    # ignore gold in this case as eval is self-contained
    score = evaluate_rag_answer(state['question'], state['answer'], state['sub_answers'])
    return {'final_output': score}
    

if __name__ == "__main__":
    from compiler.utils import load_api_key
    load_api_key('secrets.toml')
    print(evaluate_rag_answer("What are the types of agent memory?", 
                        "Agent memory in the context of artificial intelligence or autonomous agents can be categorized into several types, each serving a distinct function to enhance the agent's ability to store, retrieve, and utilize information effectively. These types include:\n\n1. **Sensory Memory**: This type of memory involves learning embedding representations for raw inputs. It acts as the initial stage where sensory data is processed and converted into a format that can be used by the agent.\n\n2. **Short-Term Memory**: Short-term memory is used for in-context learning and is limited by the finite context window length of the Transformer. It allows the agent to hold and manipulate information temporarily to make immediate decisions or actions.\n\n3. **Long-Term Memory**: Long-term memory is crucial for storing information that the agent may need to access over extended periods. It includes:\n   - **Semantic Memory**: This involves storing embedding representations of information in an external vector store, which can be accessed via fast retrieval methods like Maximum Inner Product Search (MIPS). Semantic memory helps the agent to quickly find and use relevant information.\n   - **Episodic Memory**: Episodic memory allows the agent to recall specific past experiences and events. This helps the agent to learn from previous interactions and apply that knowledge to similar future scenarios.\n   - **Procedural Memory**: Procedural memory refers to the unconscious memory of skills and routines that are performed automatically. It involves the acquisition and retrieval of skills and procedures without conscious awareness, similar to implicit memory in humans.\n\nThese different types of memory interact within an agent's architecture to create a cohesive system. Sensory memory processes raw inputs into embeddings, which are then used in short-term memory for immediate tasks. Long-term memory stores these embeddings in an external vector store, allowing the agent to retrieve relevant information efficiently using methods like MIPS and ANN algorithms. This integrated memory system enables the agent to overcome the limitations of short-term memory and make informed decisions based on past experiences and learned knowledge.", 
    [
                    "Agent memory in the context of artificial intelligence or autonomous agents refers to the mechanisms by which an agent stores and retrieves information to inform its behavior. It includes sensory memory for raw inputs, short-term memory for in-context learning, and long-term memory stored in an external vector database for fast retrieval. This memory system allows the agent to use past experiences and observations to make decisions and interact effectively.",
                    "The different types of short-term memory used by agents include sensory memory, which involves learning embedding representations for raw inputs, and short-term memory as in-context learning, which is limited by the finite context window length of the Transformer.",
                    "The different types of long-term memory used by agents are semantic memory, episodic memory, and procedural memory.",
                    "Agents use episodic memory to recall specific past experiences and events, which helps them make informed decisions and adapt to new situations. This type of memory allows them to learn from previous interactions and apply that knowledge to similar future scenarios.",
                    "Agents use semantic memory by storing embedding representations of information in an external vector store, which can be accessed via fast retrieval methods like Maximum Inner Product Search (MIPS). This allows them to overcome the limitations of finite attention spans. Common algorithms for fast MIPS include approximate nearest neighbors (ANN) to quickly find relevant information.",
                    "Procedural memory in the context of agent memory refers to the unconscious memory of skills and routines that are performed automatically. This type of memory is akin to implicit memory in humans, such as riding a bike or typing on a keyboard. It involves the acquisition and retrieval of skills and procedures without conscious awareness.",
                    "Yes, there are specialized types of memory used by agents, including sensory memory for learning embedding representations, short-term memory for in-context learning, and long-term memory as an external vector store accessible via fast retrieval methods like Maximum Inner Product Search (MIPS).",
                    "Different types of memory interact within an agent's architecture by leveraging sensory memory to create embeddings from raw inputs, which are then used in short-term memory for in-context learning within the finite context window of a Transformer. Long-term memory stores these embeddings in an external vector store, which can be accessed via fast retrieval methods like Maximum Inner Product Search (MIPS) using approximate nearest neighbors (ANN) algorithms. This setup allows the agent to overcome the limitations of short-term memory by efficiently retrieving relevant information from long-term memory."
                ]
    ))