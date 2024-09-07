from compiler.utils import load_api_key, get_bill
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda


import json

load_api_key('secrets.toml')
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

from langchain_openai.chat_models import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessage("Given a series of chess moves written in Standard Algebraic Notation (SAN), give the next move that will result in a checkmate."), 
        HumanMessagePromptTemplate.from_template("moves:\n {input}\n Give your answer in JSON format")
    ]
)

solver = chat_prompt | model
print(
    solver.invoke({"input": "1. d4 d5 2. Nf3 Nf6 3. e3 a6 4. Nc3 e6 5. Bd3 h6 6. e4 dxe4 7. Bxe4 Nxe4 8. Nxe4 Bb4+ 9. c3 Ba5 10. Qa4+ Nc6 11. Ne5 Qd5 12. f3 O-O 13. Nxc6 bxc6 14. Bf4 Ra7 15. Qb3 Qb5 16. Qxb5 cxb5 17. a4 bxa4 18. Rxa4 Bb6 19. Kf2 Bd7 20. Ke3 Bxa4 21. Ra1 Bc2 22. c4 Bxe4 23. fxe4 c5 24. d5 exd5 25. exd5 Re8+ 26. Kf3 Rae7 27. Rxa6 Bc7 28. Bd2 Re2 29. Bc3 R8e3+ 30. Kg4 Rxg2+ 31. Kf5"}).content
)


class ChessMove(BaseModel):
    """Response from the chess move task"""
    move: str = Field(
        description="The next move that will result in a checkmate"
    )
    
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic
from compiler.IR.program import Workflow, Context, hint_possible_destinations, StatePool
from compiler.IR.modules import Input, Output, CodeBox

semantic = LangChainSemantic(
    system_prompt="",
    inputs=["input"],
    output_format=ChessMove,
    following_messages=[
        HumanMessage("Given a series of chess moves written in Standard Algebraic Notation (SAN), give the next move that will result in a checkmate."), 
        HumanMessagePromptTemplate.from_template("moves:\n {input}\n")
    ]
)

qa_agent = LangChainLM('qa_agent', semantic)
qa_agent.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}

qa_flow = Workflow('qa_flow')
qa_flow.add_module(Input('start'))
qa_flow.add_module(Output('end'))
qa_flow.add_module(qa_agent)

qa_flow.add_edge('start', 'qa_agent')
qa_flow.add_edge('qa_agent', 'end')
qa_flow.compile()

state = StatePool()
state.init({
    "input": "1. d4 d5 2. Nf3 Nf6 3. e3 a6 4. Nc3 e6 5. Bd3 h6 6. e4 dxe4 7. Bxe4 Nxe4 8. Nxe4 Bb4+ 9. c3 Ba5 10. Qa4+ Nc6 11. Ne5 Qd5 12. f3 O-O 13. Nxc6 bxc6 14. Bf4 Ra7 15. Qb3 Qb5 16. Qxb5 cxb5 17. a4 bxa4 18. Rxa4 Bb6 19. Kf2 Bd7 20. Ke3 Bxa4 21. Ra1 Bc2 22. c4 Bxe4 23. fxe4 c5 24. d5 exd5 25. exd5 Re8+ 26. Kf3 Rae7 27. Rxa6 Bc7 28. Bd2 Re2 29. Bc3 R8e3+ 30. Kg4 Rxg2+ 31. Kf5", 
})
qa_flow.pregel_run(state)

print(state.news('move'))

from compiler.optimizer.params.meta_programming.mp import MetaPrompting

qa_flow.reset()
reasoning = MetaPrompting()
reasoning.apply(qa_agent)
qa_flow.compile()

qa_flow.pregel_run(state)
print(state.news('move'))