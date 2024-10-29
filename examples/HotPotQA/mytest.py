import os
from litellm import completion 
from pydantic import BaseModel

# add to env var  

messages = [{"role": "user", "content": "List 5 important events in the XIX century"}]

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent]

resp = completion(
    model="gpt-4o-mini",
    messages=messages,
    response_format=EventsList
)
print(resp)
print("Received={}".format(resp))
