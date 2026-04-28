"""Agent（智能体）

演示内容：
1. ReAct Agent（推理+行动）
2. Agent 执行过程
3. 自定义工具

运行：uv run python demos/06_agent.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from helpers import get_llm, safe_calculate

load_dotenv()


# ===== 定义工具 =====

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        "北京": "晴天，15°C",
        "上海": "多云，18°C",
        "广州": "小雨，22°C",
    }
    return weather_data.get(city, f"{city}：暂无数据")


@tool
def search(query: str) -> str:
    """搜索信息"""
    mock_data = {
        "Python": "Python 是一种高级编程语言，简洁易学",
        "LangChain": "LangChain 是 LLM 应用开发框架",
        "Agent": "Agent 是能自主决策的智能体",
    }
    for key, value in mock_data.items():
        if key.lower() in query.lower():
            return value
    return "未找到相关信息"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return safe_calculate(expression)


# ===== 演示函数 =====

def demo_bind_tools() -> None:
    """演示：绑定工具到 LLM"""
    print("=" * 50)
    print("1. 绑定工具到 LLM")
    print("=" * 50)

    llm = get_llm()
    tools = [get_weather, calculate, search]
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke("北京天气如何？")

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"调用工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")

        for t in tools:
            if t.name == tool_call["name"]:
                result = t.invoke(tool_call["args"])
                print(f"结果: {result}")
    else:
        print(f"LLM 直接回答: {response.content}")
    print()


def demo_multi_tool() -> None:
    """演示：多工具协作"""
    print("=" * 50)
    print("2. 多工具协作")
    print("=" * 50)

    llm = get_llm()
    tools = [get_weather, calculate, search]
    llm_with_tools = llm.bind_tools(tools)

    queries = [
        "北京的天气怎么样？",
        "计算 25 * 4 + 10",
        "什么是 LangChain？",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        response = llm_with_tools.invoke(query)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                for t in tools:
                    if t.name == tool_call["name"]:
                        result = t.invoke(tool_call["args"])
                        print(f"  -> {result}")
        else:
            print(f"  -> {response.content[:100]}")
    print()


def demo_tool_definition() -> None:
    """演示：工具定义的重要性"""
    print("=" * 50)
    print("3. 工具定义的重要性")
    print("=" * 50)

    print("\n工具列表：")
    for t in [get_weather, search, calculate]:
        print(f"  - {t.name}: {t.description}")
    print()

    print("Agent 选择工具的逻辑：")
    print("  1. 用户输入 -> LLM 分析意图")
    print("  2. LLM 根据工具描述选择合适工具")
    print("  3. 生成工具调用参数")
    print("  4. 执行工具并返回结果")
    print()


def main() -> None:
    print("\n" + "=" * 50)
    print("LangChain 入门：Agent（智能体）")
    print("=" * 50 + "\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    demo_bind_tools()
    demo_multi_tool()
    demo_tool_definition()

    print("=" * 50)
    print("场景六演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
