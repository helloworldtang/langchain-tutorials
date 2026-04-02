"""
场景三：Memory（对话记忆）

演示内容：
1. ConversationBufferMemory（完整记忆）
2. ConversationSummaryMemory（摘要记忆）
3. 多轮对话实现

运行：python demos/03_memory.py
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()


def demo_buffer_memory():
    """演示：ConversationBufferMemory（完整记忆）"""
    print("=" * 50)
    print("1. ConversationBufferMemory（完整记忆）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 创建带记忆的对话链
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    
    # 第一轮对话
    print("用户: 你好，我叫小明")
    response1 = conversation.predict(input="你好，我叫小明")
    print(f"AI: {response1}\n")
    
    # 第二轮对话
    print("用户: 你还记得我叫什么吗？")
    response2 = conversation.predict(input="你还记得我叫什么吗？")
    print(f"AI: {response2}\n")
    
    # 查看记忆内容
    print("记忆内容:")
    print(memory.load_memory_variables({}))
    print()


def demo_summary_memory():
    """演示：ConversationSummaryMemory（摘要记忆）"""
    print("=" * 50)
    print("2. ConversationSummaryMemory（摘要记忆）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 创建摘要记忆（适合长对话，节省 token）
    memory = ConversationSummaryMemory(llm=llm)
    
    # 模拟多轮对话
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    
    # 添加多轮对话
    conversation.predict(input="你好，我叫小明，是一名 Python 开发者")
    conversation.predict(input="我最近在学习 LangChain")
    conversation.predict(input="我对 Agent 特别感兴趣")
    
    # 查看摘要
    print("对话摘要:")
    print(memory.load_memory_variables({}))
    print()
    print("💡 优点: 适合长对话，自动总结历史，节省 token")
    print()


def demo_memory_with_lcel():
    """演示：使用 LCEL 管理记忆"""
    print("=" * 50)
    print("3. 使用 LCEL（推荐方式）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 创建聊天历史存储
    history = InMemoryChatMessageHistory()
    
    # 定义带记忆的链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # 对话函数
    def chat(input_text: str) -> str:
        # 获取历史
        history_messages = history.messages.copy()
        
        # 调用 LLM
        response = chain.invoke({
            "history": history_messages,
            "input": input_text
        })
        
        # 保存到历史
        history.add_user_message(input_text)
        history.add_ai_message(response.content)
        
        return response.content
    
    # 测试多轮对话
    print("用户: 你好，我叫小红")
    print(f"AI: {chat('你好，我叫小红')}\n")
    
    print("用户: 我喜欢编程")
    print(f"AI: {chat('我喜欢编程')}\n")
    
    print("用户: 我叫什么名字？")
    print(f"AI: {chat('我叫什么名字？')}\n")
    
    # 查看历史
    print("对话历史:")
    for msg in history.messages:
        role = "用户" if msg.type == "human" else "AI"
        print(f"  [{role}] {msg.content}")
    print()


def demo_memory_comparison():
    """演示：不同记忆类型的对比"""
    print("=" * 50)
    print("4. Memory 类型对比")
    print("=" * 50)
    
    print("""
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ Memory 类型          │ 特点                │ 适用场景            │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ BufferMemory        │ 完整记忆所有对话    │ 短对话、需要精确回顾│
│ SummaryMemory       │ 自动总结对话要点    │ 长对话、节省 token │
│ BufferWindowMemory  │ 只保留最近 N 轮     │ 上下文有限制时      │
│ VectorStoreMemory   │ 向量化存储，可检索  │ 超长对话历史        │
└─────────────────────┴─────────────────────┴─────────────────────┘

💡 选择建议：
   - 简单聊天：BufferMemory
   - 长对话：SummaryMemory
   - 有限上下文：BufferWindowMemory
   - 需要检索历史：VectorStoreMemory
    """)


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：Memory（对话记忆）")
    print("=" * 50 + "\n")
    
    # 完整记忆
    demo_buffer_memory()
    
    # 摘要记忆
    demo_summary_memory()
    
    # LCEL 方式
    demo_memory_with_lcel()
    
    # 对比
    demo_memory_comparison()
    
    print("=" * 50)
    print("✅ 场景三演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
