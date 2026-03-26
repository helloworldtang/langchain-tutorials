"""Function Call"""
from langchain import Tool
def weather(city): return {"北京": "晴15°C", "上海": "多云18°C"}.get(city, "未知")
tools = [Tool(name="weather", func=weather)]
# 简单路由
msg = "北京天气"
if "天气" in msg: print(weather(msg.replace("天气","").strip() or "北京"))
