"""MCP（Model Context Protocol）工具调用

演示内容：
1. MCP 架构介绍 + Function Call 对比
2. MCP Tools（math 服务器，stdio 连接）
3. MCP Resources & Prompts（weather 服务器，读取数据 + 加载模板）
4. MCP 多服务器 + Agent（math + weather，Agent 自动选择工具）

前置条件：
  确保 demos/mcp_math_server.py 和 demos/mcp_weather_server.py 存在

运行：uv run python demos/09_mcp.py
"""
import os
import asyncio

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent as create_react_agent
from helpers import get_llm

load_dotenv()

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
MATH_SERVER_PATH = os.path.join(DEMO_DIR, "mcp_math_server.py")
WEATHER_SERVER_PATH = os.path.join(DEMO_DIR, "mcp_weather_server.py")


# ==================== 0. 架构介绍 ====================

def print_architecture() -> None:
    print("=" * 60)
    print("MCP（Model Context Protocol）架构介绍")
    print("=" * 60)
    print("""
MCP 是 Anthropic 提出的开放协议，被称为 "AI 的 USB-C"。

架构：

  +----------+     stdio/HTTP     +----------+
  | Host     | <===============> | Server   |
  | (LLM App)|    MCP Protocol   | (FastMCP)|
  +----------+                   +----------+
       |
  +----------+
  | Client   |
  | (MCP SDK)|
  +----------+

MCP 服务器提供三大能力（Primitives）：

  +-----------+----------------------------------+------------------+
  | Primitive | 作用                             | 类比             |
  +-----------+----------------------------------+------------------+
  | Tools     | LLM 可调用的动作（函数）         | POST 接口        |
  | Resources | LLM 可读取的数据（文件/记录）    | GET 接口         |
  | Prompts   | 可复用的 Prompt 模板             | 模板引擎         |
  +-----------+----------------------------------+------------------+

vs 传统 Function Call：

  Function Call: @tool 定义 -> 绑定到 LLM -> 应用内执行
  MCP:           FastMCP 定义 -> 独立进程 -> 标准协议通信 -> 可跨应用复用
""")


# ==================== 1. Function Call ====================

@tool
def get_joke() -> str:
    """获取一条中文笑话"""
    import random
    idioms = [
        "笨鸟先飞早入林，笨人先跑早被追。",
        "失败并不可怕，可怕的是你还相信这句话。",
        "人生就像心电图，一帆风顺你就挂了。",
        "虽然你不能决定生命的长度，但你能决定它的宽度。可惜，你太宽了。",
        "世上无难事，只要肯放弃。",
        "有钱不一定幸福，没钱一定不幸福，这是废话，但很真实。"
    ]
    return f"笑话: {random.choice(idioms)}"


async def demo_function_call() -> None:
    print("=" * 60)
    print("1. Function Call（传统 @tool 方式）")
    print("=" * 60)
    print("""
传统方式：用 @tool 在应用内定义工具，绑定到 LLM。

局限：工具和 application 耦合，无法跨应用复用。
""")

    llm = get_llm()
    tools = [get_joke]
    llm_with_tools = llm.bind_tools(tools)

    print("[获取笑话...]")
    response = llm_with_tools.invoke("给我讲个笑话")

    if response.tool_calls:
        tool_name = response.tool_calls[0]["name"]
        print(f"调用工具: {tool_name}")
        result = tools[0].invoke({})
        print(f"结果: {result}")
    else:
        print(f"LLM 未调用工具: {response.content}")


# ==================== 2. MCP Tools ====================

async def demo_mcp_tools() -> None:
    print("\n" + "=" * 60)
    print("2. MCP Tools（math 服务器，stdio 连接）")
    print("=" * 60)
    print("""
MCP 方式：工具定义在独立的 MCP 服务器中（mcp_math_server.py）。

客户端通过 stdio 连接服务器，自动加载工具。
""")

    try:
        client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": [MATH_SERVER_PATH],
                "transport": "stdio",
            }
        })

        tools = await client.get_tools()

        print(f"从 MCP 服务器加载了 {len(tools)} 个工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        print("\n[手动调用工具验证]")
        for t in tools:
            if t.name == "add":
                result = await t.ainvoke({"a": 3, "b": 5})
                print(f"  add(3, 5) = {result}")
            elif t.name == "multiply":
                result = await t.ainvoke({"a": 7, "b": 8})
                print(f"  multiply(7, 8) = {result}")

    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")


# ==================== 3. MCP Resources & Prompts ====================

async def demo_mcp_resources_prompts() -> None:
    print("\n" + "=" * 60)
    print("3. MCP Resources & Prompts（weather 服务器）")
    print("=" * 60)
    print("""
MCP 不仅能提供工具（Tools），还能提供：

  - Resources：只读数据，LLM 可以获取上下文信息
  - Prompts：可复用的 Prompt 模板，带参数

weather 服务器同时提供三种能力：
  @mcp.tool()     -> get_weather(city)     查询天气
  @mcp.resource() -> weather://cities      支持的城市列表
  @mcp.prompt()   -> weather_report(city)  天气分析 Prompt
""")

    try:
        client = MultiServerMCPClient({
            "weather": {
                "command": "python",
                "args": [WEATHER_SERVER_PATH],
                "transport": "stdio",
            }
        })

        # --- Tools ---
        tools = await client.get_tools(server_name="weather")
        print(f"[Tools] 加载了 {len(tools)} 个工具:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # --- Resources ---
        print("\n[Resources] 读取服务器提供的数据:")
        resources = await client.get_resources("weather")
        for blob in resources:
            print(f"  URI: {blob.metadata.get('uri', 'unknown')}")
            print(f"  内容: {blob.data}")

        # --- Prompts ---
        print("\n[Prompts] 加载 Prompt 模板:")
        messages = await client.get_prompt(
            "weather", "weather_report",
            arguments={"city": "北京"},
        )
        for msg in messages:
            print(f"  {msg.__class__.__name__}: {msg.content[:80]}...")

    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")


# ==================== 4. MCP 多服务器 + Agent ====================

async def demo_mcp_multi_server() -> None:
    print("\n" + "=" * 60)
    print("4. MCP 多服务器 + Agent（核心场景）")
    print("=" * 60)
    print("""
MCP 的最大优势：同时连接多个独立服务器，工具自动组合。

本例同时连接：
  - math 服务器：add、multiply
  - weather 服务器：get_weather

用 create_react_agent 创建 Agent，自动选择工具回答跨域问题。
""")

    try:
        client = MultiServerMCPClient({
            "math": {
                "command": "python",
                "args": [MATH_SERVER_PATH],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": [WEATHER_SERVER_PATH],
                "transport": "stdio",
            }
        })

        tools = await client.get_tools()

        print(f"共加载 {len(tools)} 个工具（来自 2 个服务器）:")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        llm = get_llm()
        agent = create_react_agent(llm, tools)

        query = "北京和上海的天气怎么样？另外帮我算一下 (3+5)*12"
        print(f"\n问题: {query}")

        response = await agent.ainvoke({"messages": query})

        print("\nAgent 执行过程:")
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  -> 调用工具: {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                print(f"  -> 工具结果: {msg.content}")

        final = response["messages"][-1]
        print(f"\n最终回答: {final.content}")

    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")


# ==================== 对比总结 ====================

def print_comparison() -> None:
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print("""
+--------------------+---------------------------+---------------------------+
|                    | Function Call (@tool)     | MCP                       |
+--------------------+---------------------------+---------------------------+
| 工具定义           | @tool 装饰器              | FastMCP @mcp.tool()       |
| 数据提供           | 不支持                    | @mcp.resource()           |
| Prompt 模板        | 不支持                    | @mcp.prompt()             |
| 运行位置           | 应用内                    | 独立进程 / 远程服务器     |
| 多服务组合         | 不支持                    | 支持多服务器同时连接      |
| 跨应用复用         | 不支持                    | 标准协议，任何客户端可用  |
| Agent 集成         | bind_tools                | create_react_agent        |
+--------------------+---------------------------+---------------------------+

MCP 三大原语：
  Tools     -> LLM 执行动作（查询天气、计算数学）
  Resources -> LLM 获取数据（城市列表、配置信息）
  Prompts   -> LLM 使用模板（天气分析、报告生成）

什么时候用 MCP？
  - 多个服务的工具需要组合使用
  - 工具需要独立部署和升级
  - 工具需要跨应用共享（Claude Desktop / Cursor / 其他 LLM）
  - 生产环境中工具由不同团队维护
""")


async def main() -> None:
    os.environ["PYTHONUNBUFFERED"] = "1"

    try:
        print_architecture()
        await demo_function_call()
        await demo_mcp_tools()
        await demo_mcp_resources_prompts()
        await demo_mcp_multi_server()
        print_comparison()
    except Exception as e:
        print(f"\n[异常] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
