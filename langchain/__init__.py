"""
LangChain 风格框架 - 简化实现
模拟 LangChain 核心概念，使用纯 OpenAI SDK
兼容 DeepSeek 模型
"""
import os
from typing import List, Dict, Any, Callable, Optional
from openai import OpenAI


# ===== 消息类型 =====
class SystemMessage:
    """系统消息"""
    def __init__(self, content: str):
        self.content = content
        self.type = "system"

class HumanMessage:
    """用户消息"""
    def __init__(self, content: str):
        self.content = content
        self.type = "user"

class AIMessage:
    """AI 消息"""
    def __init__(self, content: str):
        self.content = content
        self.type = "ai"


# ===== LLM 封装 =====
class ChatOpenAI:
    """OpenAI 风格 LLM 封装"""
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.3,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )
    
    def invoke(self, messages: List) -> AIMessage:
        """调用 LLM"""
        # 转换消息格式
        formatted = []
        for msg in messages:
            if hasattr(msg, 'type'):
                formatted.append({"role": msg.type, "content": msg.content})
            else:
                formatted.append(msg)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=self.temperature,
        )
        return AIMessage(response.choices[0].message.content)


# ===== Tool 工具 =====
class Tool:
    """工具定义"""
    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
    
    def run(self, **kwargs) -> str:
        return self.func(**kwargs)


# ===== Agent 类型 =====
class AgentType:
    """Agent 类型常量"""
    CHAT_ZERO_SHOT_REACTING = "chat_zero_shot_reacting"
    OPENAI_FUNCTIONS = "openai_functions"


# ===== 简化版 LangChain 框架 =====
class LangChain:
    """
    简化版 LangChain 框架
    演示 LangChain 核心概念
    """
    
    def __init__(self, model: str = "deepseek-chat"):
        self.llm = ChatOpenAI(model=model)
        self.tools: Dict[str, Tool] = {}
    
    def add_tool(self, tool: Tool):
        """添加工具"""
        self.tools[tool.name] = tool
    
    def create_tools_agent(self, system_prompt: str = None):
        """创建工具 Agent"""
        return ToolAgent(self.llm, self.tools, system_prompt)


class ToolAgent:
    """工具调用 Agent"""
    
    def __init__(self, llm: ChatOpenAI, tools: Dict[str, Tool], system_prompt: str = None):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt or "你是一个有帮助的助手，可以使用工具来回答问题。"
    
    def run(self, user_input: str) -> str:
        """运行 Agent"""
        messages = [
            SystemMessage(self.system_prompt),
            HumanMessage(user_input)
        ]
        
        # 简单模拟：直接调用 LLM
        response = self.llm.invoke(messages)
        
        # 检查是否需要调用工具（简化版）
        for tool_name, tool in self.tools.items():
            if tool_name in user_input:
                try:
                    result = tool.run(query=user_input)
                    return f"{response.content}\n\n工具结果: {result}"
                except:
                    pass
        
        return response.content


# ===== 导出 =====
__all__ = [
    'SystemMessage',
    'HumanMessage', 
    'AIMessage',
    'ChatOpenAI',
    'Tool',
    'AgentType',
    'LangChain',
]
