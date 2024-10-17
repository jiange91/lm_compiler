from langchain_openai import ChatOpenAI
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

llm = ChatOpenAI(
    api_key="EMPTY",
    openai_api_base="http://192.168.1.16:30000/v1",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessage("Given a series of chess moves written in Standard Algebraic Notation (SAN), give the next move that will result in a checkmate."), 
        HumanMessagePromptTemplate.from_template("moves:\n {input}\n Give your answer in JSON format. Let's think step by step.")
    ]
)

solver = chat_prompt | llm
print(
    solver.invoke({"input": "1. d4 d5 2. Nf3 Nf6 3. e3 a6 4. Nc3 e6 5. Bd3 h6 6. e4 dxe4 7. Bxe4 Nxe4 8. Nxe4 Bb4+ 9. c3 Ba5 10. Qa4+ Nc6 11. Ne5 Qd5 12. f3 O-O 13. Nxc6 bxc6 14. Bf4 Ra7 15. Qb3 Qb5 16. Qxb5 cxb5 17. a4 bxa4 18. Rxa4 Bb6 19. Kf2 Bd7 20. Ke3 Bxa4 21. Ra1 Bc2 22. c4 Bxe4 23. fxe4 c5 24. d5 exd5 25. exd5 Re8+ 26. Kf3 Rae7 27. Rxa6 Bc7 28. Bd2 Re2 29. Bc3 R8e3+ 30. Kg4 Rxg2+ 31. Kf5"}).content
)
