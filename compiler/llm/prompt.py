from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import uuid

@dataclass
class ImageParams:
  is_image_upload: bool = False
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
  type: Literal["text"] = "text"
  text: str

@dataclass
class ImageContent:
  type: Literal["image_url"] = "image_url"
  image_url: dict
  
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
      if input_variable.is_image:
        demo_content.append(TextContent(text=demo_string))
        demo_string = ""
        if input_variable.is_image_upload:
          demo_content.append(ImageContent(image_upload=input_variable.value, file_type=input_variable.file_type))
        else:
          demo_content.append(ImageContent(image_link=input_variable.value))
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
  name: str = None
  content: List[Content]

  def __init__(self, message: Dict[str, str]):
    self.role = message['role']
    self.name = message.get('name', None)
    if isinstance(message['content'], str):
      self.content = [TextContent(text=message['content'])]
    elif isinstance(message['content'], list):
      self.content = []
      for content_entry in message['content']:
        if isinstance(content_entry, str):
          self.content.append(TextContent(text=content_entry))
        elif isinstance(content_entry, dict):
          if content_entry['type'] == 'text':
            self.content.append(TextContent(text=content_entry['text']))
          elif content_entry['type'] == 'image_url':
            self.content.append(ImageContent(image_url=content_entry['image_url']))
          else:
            raise ValueError(f"Invalid content entry type. Must be either 'text' or 'image', got {content_entry['type']}")
        else:
          raise ValueError(f"Invalid content entry. Must be either a string or a dictionary, got {type(content_entry)}")
    else:
      raise ValueError(f"Invalid content. Must be either a string or a list, got {type(message['content'])}")
    
  def to_api(self) -> Dict[str, str]:
    return {
      'role': self.role,
      'name': self.name,
      'content': [content.__dict__ for content in self.content]
    }