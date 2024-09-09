from compiler.utils import load_api_key, get_bill
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('absl').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

load_api_key('secrets.toml')
import json


from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

print(response)