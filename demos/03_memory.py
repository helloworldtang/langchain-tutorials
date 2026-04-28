"""Memory（对话记忆）

演示内容：
1. InMemoryChatMessageHistory（消息历史）
2. LCEL 方式管理记忆
3. 多轮对话实现

运行：uv run python demos/03_memory.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from helpers import get_llm

load_dotenv()


def demo_chat_history() -> None:
    """演示：对话历史管理"""
    print("=" * 50)
    print("1. InMemoryChatMessageHistory")
    print("=" * 50)

    history = InMemoryChatMessageHistory()

    # 添加消息
    history.add_user_message("你好，我叫小明")
    history.add_ai_message("你好小明！很高兴认识你！")

    print("对话历史:")
    for msg in history.messages:
        role = "用户" if msg.type == "human" else "AI"
        print(f"  [{role}] {msg.content}")
    print()


def demo_memory_with_lcel() -> None:
    """演示：使用 LCEL 管理记忆"""
    print("=" * 50)
    print("2. 使用 LCEL 管理记忆（推荐）")
    print("=" * 50)

    llm = get_llm(temperature=0.7)
    history = InMemoryChatMessageHistory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    def chat(input_text: str) -> str:
        response = chain.invoke({
            "history": history.messages,
            "input": input_text
        })
        history.add_user_message(input_text)
        history.add_ai_message(response.content)
        return response.content

    print("用户: 你好，我叫小红")
    print(f"AI: {chat('你好，我叫小红')}\n")

    print("用户: 我喜欢编程")
    print(f"AI: {chat('我喜欢编程')}\n")

    print("用户: 我叫什么名字？")
    print(f"AI: {chat('我叫什么名字？')}\n")

    print("对话历史:")
    for msg in history.messages:
        role = "用户" if msg.type == "human" else "AI"
        print(f"  [{role}] {msg.content[:50]}...")
    print()


def demo_multi_turn() -> None:
    """演示：完整多轮对话"""
    print("=" * 50)
    print("3. 完整多轮对话示例")
    print("=" * 50)

    llm = get_llm(temperature=0.7)
    history = InMemoryChatMessageHistory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 Python 专家，回答简洁专业。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    questions = [
        "什么是装饰器？",
        "能举个例子吗？",
        "装饰器有什么应用场景？"
    ]

    for q in questions:
        response = chain.invoke({
            "history": history.messages,
            "input": q
        })
        history.add_user_message(q)
        history.add_ai_message(response.content)

        print(f"用户: {q}")
        print(f"AI: {response.content[:100]}...\n")


def main() -> None:
    print("\n" + "=" * 50)
    print("LangChain 入门：Memory（对话记忆）")
    print("=" * 50 + "\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    demo_chat_history()
    demo_memory_with_lcel()
    demo_multi_turn()

    print("""
Memory 类型说明：

+---------------------+---------------------+---------------------+
| Memory 类型          | 特点                | 适用场景            |
+---------------------+---------------------+---------------------+
| InMemoryHistory     | 内存存储，简单      | 简单对话、演示      |
| Redis History       | 持久化存储          | 生产环境、多实例    |
| File History        | 文件存储            | 单机持久化          |
+---------------------+---------------------+---------------------+
""")

    print("=" * 50)
    print("场景三演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
