from typing import Literal
import concurrent.futures
from functools import wraps
import re
import os

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from compiler.langchain_bridge.interface import LangChainSemantic, LangChainLM
from compiler.IR.modules import Map, CodeBox
import base64
from plot_coder import PlottingCode


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

VISUAL_FEEDBACK_SYSTEM_PROMPT = """
You are an expert in data visualization. Given a piece of code, a user query, and an image of the current plot, please determine whether the plot has faithfully followed the user query. Your task is to provide instructions to make sure the plot has strictly completed the requirements of the query. Please output a detailed step by step instruction on how to use python code to enhance the plot.
"""

usr_prompt_template = HumanMessagePromptTemplate.from_template(
[
'''
Here is the code: [Code]:
"""
{code}
"""

Here is the user query: [Query]:
"""
{query}
"""

Carefully read and analyze the user query to understand the specific requirements. Examine the provided Python code to understand how the current plot is generated. Check if the code aligns with the user query in terms of data selection, plot type, and any specific customization. Look at the provided image of the plot. Assess the plot type, the data it represents, labels, titles, colors, and any other visual elements. Compare these elements with the requirements specified in the user query. Note any differences between the user query requirements and the current plot. Based on the identified discrepancies, provide step-by-step instructions on how to modify the Python code to meet the user query requirements. 

Also suggest improvements for better visualization practices, such as clarity, readability, and aesthetics, while ensuring the primary focus is on meeting the user's specified requirements.

Remember to save the plot to a png file. The file name should be """{plot_file_name}"""
''',
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,{plot_image}"
        }
    }
]
)

def img_encode(workspace, plot_file_name):
    base_name, extension = os.path.splitext(plot_file_name)
    plot_file_name_refine = f'{base_name}_final{extension}'
    return {'plot_image': encode_image(os.path.join(workspace, plot_file_name)), 
            'plot_file_name': plot_file_name_refine}

img_encode_codebox = CodeBox('img encode', img_encode)

class VisualRefinementSchema(BaseModel):
    """Response from the visual refinement task"""
    visual_refinement: str = Field(
        description="details on how to enhance the plot"
    )
    
visual_refinement_semantic = LangChainSemantic(
    VISUAL_FEEDBACK_SYSTEM_PROMPT,
    ['query', 'code', 'plot_image', 'plot_file_name'],
    "visual_refinement",
    following_messages=[usr_prompt_template],
    img_input_idices=[2],
)

visual_refinement = LangChainLM('visual refinement', visual_refinement_semantic)


VIS_SYSTEM_PROMPT = """
You are a cutting-edge super capable code generation LLM. You will be given a piece of code and natural language instruction on how to improve it. Base on the given code, generate a runnable python code to satisfy all the requirements in the instruction while retaining the original code's functionality. You can use any python library you want. 

In your code, when you complete a plot, remember to save it to a png file with given the 'plot_file_name'.
"""


refine_semantic = LangChainSemantic(
    system_prompt=VIS_SYSTEM_PROMPT,
    inputs=['code', 'visual_refinement', 'plot_file_name'],
    output_format="code",
    output_format_instructions="Please only return the python code. Wrap it with ```python and ``` to format it properly.",
)

refine_plot_coder = LangChainLM('visual refine coder', refine_semantic)