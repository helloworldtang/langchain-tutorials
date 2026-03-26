"""LLM 直接调用"""
from langchain import ChatOpenAI, HumanMessage
llm = ChatOpenAI(model="deepseek-chat")
print(llm.invoke([HumanMessage("你好")]).content)
