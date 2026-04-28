"""Function Call（工具调用）

演示内容：
1. 定义工具（@tool 装饰器）
2. 绑定工具到 LLM
3. 自动执行工具调用

运行：uv run python demos/02_function_call.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from helpers import get_llm, safe_calculate

load_dotenv()


# ===== 定义工具 =====

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气

    Args:
        city: 城市名称，如"北京"、"上海"

    Returns:
        天气信息字符串
    """
    weather_data = {
        "北京": "晴天，温度 15°C，空气质量良好",
        "上海": "多云，温度 18°C，有轻微雾霾",
        "广州": "小雨，温度 22°C，湿度较高",
        "深圳": "阴天，温度 23°C，适合出行",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式，如"2+3*4"

    Returns:
        计算结果
    """
    return safe_calculate(expression)


@tool
def search_database(query: str) -> str:
    """搜索数据库"""
    mock_data = {
        "用户": "共 1,234 名活跃用户，今日新增 56 名",
        "订单": "今日订单 89 笔，总金额 ¥12,345",
        "产品": "在售产品 45 个，库存充足",
    }
    for key, value in mock_data.items():
        if key in query:
            return value
    return "未找到相关数据"


# ===== 演示函数 =====

def demo_bind_tools() -> None:
    """演示：绑定工具到 LLM"""
    print("=" * 50)
    print("1. 绑定工具到 LLM")
    print("=" * 50)

    llm = get_llm()
    tools = [get_weather, calculate, search_database]
    llm_with_tools = llm.bind_tools(tools)

    print("\n[查询天气...]")
    response = llm_with_tools.invoke("北京今天天气怎么样？")

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"LLM 决定调用工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
    else:
        print(f"LLM 直接回答: {response.content}")
    print()


def demo_manual_tool_execution() -> None:
    """演示：手动执行工具调用"""
    print("=" * 50)
    print("2. 手动执行工具调用（推荐方式）")
    print("=" * 50)

    llm = get_llm()
    tools = [get_weather, calculate, search_database]
    llm_with_tools = llm.bind_tools(tools)

    queries = [
        "上海和广州的天气",
        "计算 123 * 456",
    ]

    for query in queries:
        print(f"\n查询: {query}")
        response = llm_with_tools.invoke(query)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                print(f"  调用: {tool_name}({args})")

                for t in tools:
                    if t.name == tool_name:
                        result = t.invoke(args)
                        print(f"  结果: {result}")
        else:
            print(f"  回答: {response.content}")
    print()


def demo_tool_description() -> None:
    """演示：工具描述的重要性"""
    print("=" * 50)
    print("3. 工具描述的重要性")
    print("=" * 50)

    print("\n工具列表：")
    for t in [get_weather, calculate, search_database]:
        print(f"  - {t.name}: {t.description[:30]}...")
    print()


def main() -> None:
    print("\n" + "=" * 50)
    print("LangChain 入门：Function Call（工具调用）")
    print("=" * 50 + "\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    demo_bind_tools()
    demo_manual_tool_execution()
    demo_tool_description()

    print("=" * 50)
    print("场景二演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
