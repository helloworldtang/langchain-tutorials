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

load_dotenv()


def get_llm(temperature=0):
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature
    )


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
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool  
def search_database(query: str) -> str:
    """
    在数据库中搜索信息
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果
    """
    mock_db = {
        "用户": "共 1000 个用户，其中活跃用户 800 个",
        "订单": "今日订单 150 个，总金额 50000 元",
        "商品": "共 500 个商品，库存充足",
    }
    
    for key, value in mock_db.items():
        if key in query:
            return value
    return "未找到相关信息"


# ===== 演示函数 =====

def demo_bind_tools():
    """演示：绑定工具到 LLM"""
    print("=" * 50)
    print("1. 绑定工具到 LLM")
    print("=" * 50)
    
    llm = get_llm()
    tools = [get_weather, calculate, search_database]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke("北京今天天气怎么样？")
    
    print(f"回复内容: {response.content}")
    print(f"工具调用: {response.tool_calls}")
    print()


def demo_manual_tool_execution():
    """演示：手动执行工具"""
    print("=" * 50)
    print("2. 手动执行工具调用")
    print("=" * 50)
    
    llm = get_llm()
    tools = [get_weather, calculate]
    llm_with_tools = llm.bind_tools(tools)
    
    user_input = "北京和上海今天天气怎么样？"
    response = llm_with_tools.invoke(user_input)
    
    if response.tool_calls:
        print(f"LLM 决定调用 {len(response.tool_calls)} 个工具:")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  - 工具: {tool_name}")
            print(f"    参数: {tool_args}")
            
            if tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
                print(f"    结果: {result}")
    print()


def demo_tool_description():
    """演示：工具描述的重要性"""
    print("=" * 50)
    print("3. 工具描述的重要性")
    print("=" * 50)
    
    print("get_weather 工具描述:")
    print(f"  {get_weather.description}")
    print()
    print("calculate 工具描述:")
    print(f"  {calculate.description}")
    print()
    print("💡 提示: 工具描述要清晰，说明：")
    print("  1. 工具的功能")
    print("  2. 参数的含义")
    print("  3. 返回的内容")
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：Function Call（工具调用）")
    print("=" * 50 + "\n")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    demo_bind_tools()
    demo_manual_tool_execution()
    demo_tool_description()
    
    print("=" * 50)
    print("✅ 场景二演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
