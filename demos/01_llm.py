"""
场景一：LLM 直接调用

演示内容：
1. 基本的 LLM 调用
2. 使用环境变量配置 API Key
3. 国产模型（DeepSeek）调用方式

运行：python demos/01_llm.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 加载环境变量
load_dotenv()


def demo_basic_call():
    """基本调用示例"""
    print("=" * 50)
    print("1. 基本 LLM 调用")
    print("=" * 50)
    
    # 创建 LLM 实例（自动从环境变量读取 OPENAI_API_KEY）
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # 调用
    response = llm.invoke("你好，请用一句话介绍你自己")
    print(f"回复: {response.content}")
    print()


def demo_with_system_prompt():
    """带系统提示词的调用"""
    print("=" * 50)
    print("2. 带系统提示词的调用")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    messages = [
        SystemMessage(content="你是一个专业的 Python 讲师，回答要简洁、实用。"),
        HumanMessage(content="什么是装饰器？")
    ]
    
    response = llm.invoke(messages)
    print(f"回复: {response.content}")
    print()


def demo_deepseek():
    """使用 DeepSeek 模型"""
    print("=" * 50)
    print("3. 使用 DeepSeek 模型")
    print("=" * 50)
    
    # DeepSeek 兼容 OpenAI API，只需修改 base_url
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY", "your-deepseek-key")
    )
    
    response = llm.invoke("你好")
    print(f"回复: {response.content}")
    print()


def demo_streaming():
    """流式输出"""
    print("=" * 50)
    print("4. 流式输出（逐字打印）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)
    
    print("回复: ", end="", flush=True)
    for chunk in llm.stream("用三句话介绍 Python"):
        print(chunk.content, end="", flush=True)
    print("\n")


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：LLM 直接调用")
    print("=" * 50 + "\n")
    
    # 基本调用
    demo_basic_call()
    
    # 带系统提示词
    demo_with_system_prompt()
    
    # 流式输出
    demo_streaming()
    
    # DeepSeek（需要配置 DEEPSEEK_API_KEY）
    # demo_deepseek()
    
    print("=" * 50)
    print("✅ 场景一演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
