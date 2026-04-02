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
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算错误"


# ===== 演示函数 =====

def demo_bind_tools():
    """演示：绑定工具到 LLM"""
    print("=" * 50)
    print("1. 绑定工具到 LLM")
    print("=" * 50)
    
    llm = get_llm()
    tools = [get_weather, calculate]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke("北京今天天气怎么样？")
    
    print(f"回复内容: {response.content[:100]}...")
    print(f"工具调用: {response.tool_calls}")
    print()


def demo_multi_tool():
    """演示：多工具调用"""
    print("=" * 50)
    print("2. 多工具调用分析")
    print("=" * 50)
    
    llm = get_llm()
    tools = [get_weather, calculate]
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke("北京今天天气怎么样？顺便帮我算一下 25 * 4")
    
    print(f"工具调用数量: {len(response.tool_calls) if response.tool_calls else 0}")
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  - {tc['name']}: {tc['args']}")
    print()


def demo_tool_definition():
    """演示：工具定义"""
    print("=" * 50)
    print("3. 工具定义示例")
    print("=" * 50)
    
    print("定义工具：")
    print("""
@tool
def get_weather(city: str) -> str:
    \"\"\"查询城市天气\"\"\"
    weather_data = {
        "北京": "晴天，15°C",
        "上海": "多云，18°C",
    }
    return weather_data.get(city, f"{city}：暂无数据")
""")
    
    print("绑定工具：")
    print("""
llm = ChatOpenAI(model="deepseek-chat", ...)
tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)
""")
    
    print("💡 关键点：")
    print("  1. 工具描述决定了 LLM 何时调用")
    print("  2. 参数类型提示帮助 LLM 生成正确参数")
    print("  3. 工具数量建议控制在 10 个以内")
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：Agent（智能体）")
    print("=" * 50 + "\n")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    demo_bind_tools()
    demo_multi_tool()
    demo_tool_definition()
    
    print("""
Agent 核心概念：

┌─────────────────────────────────────────────────────────────┐
│                    Agent 工作流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户问题："北京天气怎么样？"                               │
│         ↓                                                   │
│   LLM 分析：需要调用天气工具                                 │
│         ↓                                                   │
│   返回工具调用：get_weather(city="北京")                    │
│         ↓                                                   │
│   执行工具，返回结果                                         │
│         ↓                                                   │
│   LLM 整理答案："北京今天晴天，15°C"                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

💡 Agent vs Function Call：
   - Function Call：手动处理工具调用
   - Agent：自动决策、自动调用、自动整理答案
""")
    
    print("=" * 50)
    print("✅ 场景六演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
