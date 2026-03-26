"""MCP 协议模拟"""
class MCPServer:
    def __init__(self): self.tools = {}
    def register(self, name, func): self.tools[name] = func
    def call(self, name, args): return self.tools.get(name, lambda: "未找到")(**args)
server = MCPServer()
server.register("search", lambda q: f"结果: {q}")
print(server.call("search", {"q": "test"}))
