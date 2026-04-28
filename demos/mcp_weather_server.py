"""MCP 天气服务器（演示 Tools + Resources + Prompts）

提供天气查询工具、城市列表资源、天气分析 Prompt。

运行方式：
  uv run python demos/mcp_weather_server.py  # 直接运行（测试用）

客户端连接（stdio transport）：
  MultiServerMCPClient({
      "weather": {
          "command": "python",
          "args": ["demos/mcp_weather_server.py"],
          "transport": "stdio",
      }
  })
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

WEATHER_DATA = {
    "北京": "晴天，15°C，空气质量良好",
    "上海": "多云，18°C，有轻微雾霾",
    "广州": "小雨，22°C，湿度较高",
    "深圳": "阴天，23°C，适合出行",
}


# ===== Tools：LLM 可调用的动作 =====

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city"""
    return WEATHER_DATA.get(city, f"{city}：暂无天气数据")


# ===== Resources：LLM 可读取的数据 =====

@mcp.resource("weather://cities")
def get_supported_cities() -> str:
    """List supported cities for weather queries"""
    return "支持的城市：北京、上海、广州、深圳"


# ===== Prompts：可复用的 Prompt 模板 =====

@mcp.prompt()
def weather_report(city: str) -> list[dict]:
    """Generate a weather analysis prompt for a city"""
    weather = WEATHER_DATA.get(city, "未知")
    return [
        {
            "role": "user",
            "content": (
                f"以下是 {city} 的天气数据：{weather}\n"
                f"请分析天气状况，并给出出行建议。"
            ),
        }
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
