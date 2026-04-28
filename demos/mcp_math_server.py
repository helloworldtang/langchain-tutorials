"""MCP 数学工具服务器

提供加法和乘法工具。

运行方式：
  uv run python demos/mcp_math_server.py  # 直接运行（测试用）

客户端连接（stdio transport）：
  MultiServerMCPClient({
      "math": {
          "command": "python",
          "args": ["demos/mcp_math_server.py"],
          "transport": "stdio",
      }
  })
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
