from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import uuid

@dataclass
class ImageParams:
  is_image_upload: bool
  file_type: Literal['jpeg', 'png']

@dataclass
class InputVar:
  name: str
  image_params: Optional[ImageParams] = None

@dataclass
class FilledInputVar:
  input_variable: InputVar
  value: str

@dataclass
class TextContent:
  text: str
  type: Literal["text"] = "text"

@dataclass
class ImageContent:
  image_url: dict
  type: Literal["image_url"] = "image_url"
  
  def __init__(self, image_link: str):
    self.image_url = {"url": image_link}

  def __init__(self, image_upload: str, file_type: Literal['jpeg', 'png']):
    self.image_url = {"url": f"data:image/{file_type};base64,{image_upload}"}

Content = TextContent | ImageContent

@dataclass
class Demonstration:
  filled_input_variables: List[FilledInputVar]
  output: str
  id: str
  reasoning: str = None

  def __init__(self, inputs: Dict[str, str], output: str, id: str = None, reasoning: str = None):
    self.filled_input_variables = [FilledInputVar(input_variable=InputVar(name=key), value=value) for key, value in inputs.items()]
    self.output = output
    self.id = id or str(uuid.uuid4())
    self.reasoning = reasoning

  def to_content(self) -> List[Content]:
    demo_content: List[Content] = []
    demo_string = ""
    demo_string = "**Input:**\n"
    for filled in self.filled_input_variables:
      demo_string += f'{filled.input_variable.name}:\n'
      input_variable = filled.input_variable
      if input_variable.image_params:
        demo_content.append(TextContent(text=demo_string))
        demo_string = ""
        if input_variable.image_params.is_image_upload:
          demo_content.append(ImageContent(image_upload=filled.value, file_type=input_variable.image_params.file_type))
        else:
          demo_content.append(ImageContent(image_link=filled.value))
    if self.reasoning is not None:
      demo_string += f'**Reasoning:**\n{self.reasoning}\n'
    else:
      demo_string += f'**Reasoning:**\nnot available'
    demo_string += f'**Answer:**\n{self.output}'
    demo_content.append(TextContent(text=demo_string))
    return demo_content

@dataclass
class CompletionMessage:
  role: Literal['system', 'user', 'assistant']
  content: List[Content]
  name: str = None

  def to_api(self) -> Dict[str, str]:
    return {
      'role': self.role,
      'name': self.name,
      'content': [content.__dict__ for content in self.content]
    }
