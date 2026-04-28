"""LLM 直接调用

演示内容：
1. 基本 LLM 调用
2. 流式输出
3. 带系统提示词的对话

运行：uv run python demos/01_llm.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from helpers import get_llm, DEEPSEEK_API_BASE

load_dotenv()


def demo_basic_call() -> None:
    """基本调用示例"""
    print("=" * 50)
    print("1. 基本 LLM 调用")
    print("=" * 50)

    llm = get_llm(temperature=0.7)
    response = llm.invoke("你好，请用一句话介绍你自己")
    print(f"回复: {response.content}")
    print()


def demo_with_system_prompt() -> None:
    """带系统提示词的调用"""
    print("=" * 50)
    print("2. 带系统提示词的调用")
    print("=" * 50)

    llm = get_llm(temperature=0.7)

    messages = [
        SystemMessage(content="你是一个专业的 Python 讲师，回答要简洁、实用。"),
        HumanMessage(content="什么是装饰器？")
    ]

    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def demo_streaming() -> None:
    """流式输出"""
    print("=" * 50)
    print("3. 流式输出（逐字打印）")
    print("=" * 50)

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base=DEEPSEEK_API_BASE,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.7,
        streaming=True
    )

    print("回复: ", end="", flush=True)
    for chunk in llm.stream("用三句话介绍 Python"):
        print(chunk.content, end="", flush=True)
    print("\n")


def main() -> None:
    print("\n" + "=" * 50)
    print("LangChain 入门：LLM 直接调用")
    print("=" * 50 + "\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        print("   cp .env.example .env")
        print("   然后编辑 .env 文件填入你的 API Key")
        return

    demo_basic_call()
    demo_with_system_prompt()
    demo_streaming()

    print("=" * 50)
    print("场景一演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
