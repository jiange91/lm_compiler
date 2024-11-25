from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
import dotenv
dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=300)

interpreter_prompt = """
You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
"""

interpreter_template = ChatPromptTemplate.from_messages(
    [
        ("system", interpreter_prompt),
        ("human", "problem:\n{problem}\n"),
    ]
)

interpreter_agent = interpreter_template | model

solver_prompt = """
You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
"""

solver_template = ChatPromptTemplate.from_messages(
    [
        ("system", solver_prompt),
        ("human", "problem:\n{problem}\n\nmath model:\n{math_model}\n"),
    ]
)

solver_agent = solver_template | model

import cognify

# Define Workflow
@cognify.register_workflow
def math_solver_workflow(problem):
    math_model = interpreter_agent.invoke({"problem": problem}).content
    answer = solver_agent.invoke({"problem": problem, "math_model": math_model}).content
    return {"answer": answer}


if __name__ == "__main__":
    problem = "A bored student walks down a hall that contains a row of closed lockers, numbered $1$ to $1024$. He opens the locker numbered 1, and then alternates between skipping and opening each locker thereafter. When he reaches the end of the hall, the student turns around and starts back. He opens the first closed locker he encounters, and then alternates between skipping and opening each closed locker thereafter. The student continues wandering back and forth in this manner until every locker is open. What is the number of the last locker he opens?\n"
    solution = "On his first pass, he opens all of the odd lockers. So there are only even lockers closed. Then he opens the lockers that are multiples of $4$, leaving only lockers $2 \\pmod{8}$ and $6 \\pmod{8}$. Then he goes ahead and opens all lockers $2 \\pmod {8}$, leaving lockers either $6 \\pmod {16}$ or $14 \\pmod {16}$. He then goes ahead and opens all lockers $14 \\pmod {16}$, leaving the lockers either $6 \\pmod {32}$ or $22 \\pmod {32}$. He then goes ahead and opens all lockers $6 \\pmod {32}$, leaving $22 \\pmod {64}$ or $54 \\pmod {64}$. He then opens $54 \\pmod {64}$, leaving $22 \\pmod {128}$ or $86 \\pmod {128}$. He then opens $22 \\pmod {128}$ and leaves $86 \\pmod {256}$ and $214 \\pmod {256}$. He then opens all $214 \\pmod {256}$, so we have $86 \\pmod {512}$ and $342 \\pmod {512}$, leaving lockers $86, 342, 598$, and $854$, and he is at where he started again. He then opens $86$ and $598$, and then goes back and opens locker number $854$, leaving locker number $\\boxed{342}$ untouched. He opens that locker."
    answer = math_solver_workflow(problem)
    print(answer)
    