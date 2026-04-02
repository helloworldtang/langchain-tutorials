"""
场景六：Agent（智能体）

演示内容：
1. ReAct Agent（推理+行动）
2. Agent 执行过程
3. 自定义工具

运行：python demos/06_agent.py
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

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
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算错误"


# ===== 演示函数 =====

def demo_basic_agent():
    """演示：基本 Agent"""
    print("=" * 50)
    print("1. 基本 Agent")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, search, calculate]
    
    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    # 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 执行
    result = agent_executor.invoke({"input": "北京今天天气怎么样？"})
    print(f"\n最终答案: {result['output']}\n")


def demo_multi_tool():
    """演示：多工具调用"""
    print("=" * 50)
    print("2. 多工具调用")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, calculate]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具回答问题。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 这个问题需要调用多个工具
    result = agent_executor.invoke({
        "input": "北京今天天气怎么样？顺便帮我算一下 25 * 4"
    })
    print(f"\n最终答案: {result['output']}\n")


def demo_agent_with_memory():
    """演示：带记忆的 Agent"""
    print("=" * 50)
    print("3. 带记忆的 Agent")
    print("=" * 50)
    
    from langchain.memory import ConversationBufferMemory
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather, search]
    
    # 创建记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，可以使用工具回答问题。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=True
    )
    
    # 第一轮
    print("用户: 我叫小明")
    result1 = agent_executor.invoke({"input": "我叫小明"})
    print(f"AI: {result1['output']}\n")
    
    # 第二轮
    print("用户: 我叫什么名字？")
    result2 = agent_executor.invoke({"input": "我叫什么名字？"})
    print(f"AI: {result2['output']}\n")


def demo_error_handling():
    """演示：错误处理"""
    print("=" * 50)
    print("4. 错误处理")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        max_iterations=3,           # 最大迭代次数
        handle_parsing_errors=True  # 处理解析错误
    )
    
    result = agent_executor.invoke({"input": "你好"})
    print(f"答案: {result['output']}\n")


def demo_agent_configuration():
    """演示：Agent 配置项"""
    print("=" * 50)
    print("5. Agent 配置项")
    print("=" * 50)
    
    print("""
AgentExecutor 关键配置：

┌─────────────────────┬─────────────────────────────────────┐
│ 参数                │ 说明                                │
├─────────────────────┼─────────────────────────────────────┤
│ max_iterations      │ 最大迭代次数（防止死循环）          │
│ max_execution_time  │ 最大执行时间（秒）                  │
│ early_stopping_     │ 超时后的行为：generate/force        │
│ method              │                                     │
│ handle_parsing_     │ 是否处理解析错误                    │
│ errors              │                                     │
│ verbose             │ 是否打印执行过程                    │
└─────────────────────┴─────────────────────────────────────┘

💡 最佳实践：
   - 设置 max_iterations=5（防止无限循环）
   - 设置 handle_parsing_errors=True（更健壮）
   - 生产环境关闭 verbose=True
""")


# 修复导入
from langchain_core.prompts import MessagesPlaceholder


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：Agent（智能体）")
    print("=" * 50 + "\n")
    
    # 基本 Agent
    demo_basic_agent()
    
    # 多工具
    demo_multi_tool()
    
    # 错误处理
    demo_error_handling()
    
    # 配置项
    demo_agent_configuration()
    
    print("=" * 50)
    print("✅ 场景六演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
