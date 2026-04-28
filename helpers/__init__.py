"""公共工具和通用函数"""
import os
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base=DEEPSEEK_API_BASE,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embeddings 实例（使用 OpenAI）"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def safe_calculate(expression: str) -> str:
    """安全计算数学表达式（仅允许数字和基本运算符）

    Args:
        expression: 数学表达式，如 "2+3*4"

    Returns:
        计算结果字符串
    """
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%]+$', sanitized):
        return "计算错误: 表达式包含非法字符"
    if re.search(r'\*\*|//', sanitized):
        return "计算错误: 不支持幂运算或整除"
    try:
        result = eval(sanitized)  # noqa: S307 — 已通过正则白名单校验
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"
