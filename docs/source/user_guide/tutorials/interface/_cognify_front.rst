.. code-block:: python

    import cognify

    interpreter_prompt = """
    You are a math problem interpreter. Your task is to analyze the problem, identify key variables, and formulate the appropriate mathematical model or equation needed to solve it. Be concise and clear in your response.
    """
    interpreter_agent = cognify.Model(
        "interpreter", 
        system_prompt=interpreter_prompt, 
        input_variables=[
            cognify.Input("problem")
        ], 
        output=cognify.OutputLabel("math_model"),
        lm_config=cognify.LMConfig(model="gpt-4o-mini")
    )

    from pydantic import BaseModel
    class MathResponse(BaseModel):
        final_answer: float
        explanation: str

    solver_prompt = """
    You are a math solver. Given a math problem, and a mathematical model for solving it, your task is to compute the solution and return the final answer. Be concise and clear in your response.
    """
    solver_agent = cognify.StructuredModel(
        "solver",
        system_prompt=solver_prompt,
        input_variables=[
            cognify.Input("problem"), 
            cognify.Input("math_model")
        ],
        output_format=cognify.OutputFormat(MathResponse),
        lm_config=cognify.LMConfig(model="gpt-4o-mini")
    )

    # Define Workflow
    def math_solver_workflow(problem):
        math_model = interpreter_agent(inputs={"problem": problem})
        response: MathResponse = solver_agent(inputs={"problem": problem, "math_model": math_model})
    return response.final_answer