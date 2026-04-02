"""LangGraph（多 Agent 协作）

演示内容：
1. 状态图基础
2. 节点定义
3. 多 Agent 协作流程

运行：uv run python demos/07_langgraph.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from operator import add

load_dotenv()


def get_llm(temperature=0):
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature
    )


# ===== 定义状态 =====

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add]
    next_agent: str


# ===== 定义工具 =====

@tool
def search_tool(query: str) -> str:
    """搜索工具"""
    data = {
        "Python": "Python 是一种高级编程语言",
        "LangChain": "LangChain 是 LLM 应用框架",
        "天气": "北京今天晴天，15°C",
    }
    for key, value in data.items():
        if key in query:
            return value
    return "未找到相关信息"


# ===== 定义 Agent 节点 =====

def researcher_node(state: AgentState) -> AgentState:
    """研究员节点：负责收集信息"""
    llm = get_llm()
    
    messages = state["messages"]
    last_message = messages[-1] if messages else HumanMessage(content="")
    
    if len(messages) == 1:
        system_msg = HumanMessage(content="你是一个研究员，负责收集信息。请简洁回答用户的问题。")
        response = llm.invoke([system_msg, last_message])
    else:
        response = llm.invoke(messages)
    
    return {"messages": [response], "next_agent": "analyst"}


def analyst_node(state: AgentState) -> AgentState:
    """分析师节点：负责分析数据"""
    llm = get_llm()
    
    messages = state["messages"]
    analysis_prompt = HumanMessage(content="请对以上研究结果进行简要分析。")
    
    response = llm.invoke(messages + [analysis_prompt])
    
    return {"messages": [response], "next_agent": "writer"}


def writer_node(state: AgentState) -> AgentState:
    """写作者节点：负责生成最终报告"""
    llm = get_llm()
    
    messages = state["messages"]
    write_prompt = HumanMessage(content="请基于以上内容，写一份简洁的总结（不超过100字）。")
    
    response = llm.invoke(messages + [write_prompt])
    
    return {"messages": [response], "next_agent": "end"}


# ===== 演示函数 =====

def demo_simple_workflow():
    """演示：简单工作流"""
    print("=" * 50)
    print("1. 简单工作流（研究员 → 分析师 → 写作者）")
    print("=" * 50)
    
    # 创建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    # 定义边
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)
    
    # 编译
    app = workflow.compile()
    
    # 执行
    initial_state = {
        "messages": [HumanMessage(content="什么是 LangChain？")],
        "next_agent": "researcher"
    }
    
    result = app.invoke(initial_state)
    
    print(f"最终消息数: {len(result['messages'])}")
    print(f"最后一条消息: {result['messages'][-1].content}")
    print()


def demo_graph_visualization():
    """演示：图的描述"""
    print("=" * 50)
    print("2. LangGraph 架构图")
    print("=" * 50)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 工作流                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐                                               │
│   │  用户   │                                               │
│   └────┬────┘                                               │
│        ↓                                                    │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│   │研究员   │ ──→ │分析师   │ ──→ │写作者   │             │
│   │researcher│    │analyst  │     │writer   │             │
│   └─────────┘     └─────────┘     └─────────┘             │
│                                         │                   │
│                                         ↓                   │
│                                    ┌─────────┐              │
│                                    │  END    │              │
│                                    └─────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

核心概念：
1. StateGraph：状态图，管理节点间的数据流转
2. Node：节点，执行具体任务的函数
3. Edge：边，定义节点间的流转关系
4. Conditional Edges：条件边，根据状态决定下一节点

💡 LangGraph vs 串行调用：
   - 串行调用：代码耦合，难以扩展
   - LangGraph：状态管理清晰，支持循环、条件分支
""")


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：LangGraph（多 Agent 协作）")
    print("=" * 50 + "\n")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    demo_simple_workflow()
    demo_graph_visualization()
    
    print("=" * 50)
    print("✅ 场景七演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
