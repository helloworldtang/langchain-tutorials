"""
场景二：Function Call（工具调用）

演示内容：
1. 定义工具（@tool 装饰器）
2. 绑定工具到 LLM
3. 自动执行工具调用

运行：python demos/02_function_call.py
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()


# ===== 第一步：定义工具 =====

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的天气
    
    Args:
        city: 城市名称，如"北京"、"上海"
    
    Returns:
        天气信息字符串
    """
    # 模拟天气数据（实际应调用天气 API）
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
        # 注意：实际生产环境应使用更安全的表达式解析
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
    # 模拟数据库
    mock_db = {
        "用户": "共 1000 个用户，其中活跃用户 800 个",
        "订单": "今日订单 150 个，总金额 50000 元",
        "商品": "共 500 个商品，库存充足",
    }
    
    for key, value in mock_db.items():
        if key in query:
            return value
    return "未找到相关信息"


# ===== 第二步：工具调用演示 =====

def demo_bind_tools():
    """演示：绑定工具到 LLM"""
    print("=" * 50)
    print("1. 绑定工具到 LLM")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 绑定工具
    tools = [get_weather, calculate, search_database]
    llm_with_tools = llm.bind_tools(tools)
    
    # 调用
    response = llm_with_tools.invoke("北京今天天气怎么样？")
    
    print(f"回复内容: {response.content}")
    print(f"工具调用: {response.tool_calls}")
    print()


def demo_manual_tool_execution():
    """演示：手动执行工具"""
    print("=" * 50)
    print("2. 手动执行工具调用")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate]
    llm_with_tools = llm.bind_tools(tools)
    
    # 用户提问
    user_input = "北京和上海今天天气怎么样？"
    response = llm_with_tools.invoke(user_input)
    
    # 检查是否需要调用工具
    if response.tool_calls:
        print(f"LLM 决定调用 {len(response.tool_calls)} 个工具:")
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  - 工具: {tool_name}")
            print(f"    参数: {tool_args}")
            
            # 执行工具
            if tool_name == "get_weather":
                result = get_weather.invoke(tool_args)
                print(f"    结果: {result}")
    print()


def demo_auto_tool_agent():
    """演示：自动执行工具的 Agent"""
    print("=" * 50)
    print("3. 自动执行工具的 Agent")
    print("=" * 50)
    
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate, search_database]
    
    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具来回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 执行
    result = agent_executor.invoke({"input": "北京今天天气怎么样？顺便帮我算一下 25*4"})
    print(f"\n最终答案: {result['output']}")
    print()


def demo_tool_description_importance():
    """演示：工具描述的重要性"""
    print("=" * 50)
    print("4. 工具描述的重要性")
    print("=" * 50)
    
    # 工具描述决定了 LLM 何时调用该工具
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
    
    # 演示绑定工具
    demo_bind_tools()
    
    # 演示手动执行
    demo_manual_tool_execution()
    
    # 演示自动 Agent
    demo_auto_tool_agent()
    
    # 工具描述重要性
    demo_tool_description_importance()
    
    print("=" * 50)
    print("✅ 场景二演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
