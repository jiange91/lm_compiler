#================================================================
# Evaluator
#================================================================

import cognify

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@cognify.register_evaluator
def evaluate(problem, answer, solution):
    evaluator_prompt = """
You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

You should not solve the problem by yourself, a standard solution will be provided. 

Please only respond with the score number, which should be a number between 0 and 10. No additional text is needed.
    """
    evaluator_template = ChatPromptTemplate.from_messages(
        [
            ("system", evaluator_prompt),
            ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n"),
        ]
    )
    evaluator_agent = evaluator_template | model
    score = evaluator_agent.invoke({"problem": problem, "answer": answer, "solution": solution}).content
    return int(score)


#================================================================
# Data Loader
#================================================================

import json

@cognify.register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = {
            'problem': d["problem"],
        }
        ground_truth = {
            'solution': d["solution"],
        }
        new_data.append((input, ground_truth))
    return new_data[:5], None, new_data[:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

search_settings = default.create_search(
    n_trials=5,
)