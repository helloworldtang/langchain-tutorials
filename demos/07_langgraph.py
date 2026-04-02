"""
场景七：LangGraph（多 Agent 协作）

演示内容：
1. 状态图基础
2. 条件路由
3. 多 Agent 协作

运行：python demos/07_langgraph.py
"""
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from operator import add

load_dotenv()


# ===== 定义状态 =====

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add]  # 消息历史
    next_agent: str  # 下一个要执行的 Agent


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


@tool
def analysis_tool(data: str) -> str:
    """分析工具"""
    return f"分析结果：{data} 的关键信息已提取"


@tool
def summary_tool(content: str) -> str:
    """总结工具"""
    return f"总结：{content[:50]}..."


# ===== 定义 Agent 节点 =====

def researcher_node(state: AgentState) -> AgentState:
    """研究员节点：负责收集信息"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_tool]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state["messages"]
    last_message = messages[-1] if messages else HumanMessage(content="")
    
    # 如果是首次调用，添加系统提示
    if len(messages) == 1:
        system_msg = HumanMessage(content="你是一个研究员，负责收集信息。请搜索并回答用户的问题。")
        response = llm_with_tools.invoke([system_msg, last_message])
    else:
        response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "next_agent": "analyst"}


def analyst_node(state: AgentState) -> AgentState:
    """分析师节点：负责分析数据"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 获取之前的研究结果
    messages = state["messages"]
    
    # 添加分析指令
    analysis_prompt = HumanMessage(
        content="请对以上研究结果进行分析，提取关键信息。"
    )
    
    response = llm.invoke(messages + [analysis_prompt])
    
    return {"messages": [response], "next_agent": "writer"}


def writer_node(state: AgentState) -> AgentState:
    """写作者节点：负责生成最终报告"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = state["messages"]
    
    # 添加写作指令
    write_prompt = HumanMessage(
        content="请基于以上分析和研究，写一份简洁的总结报告。"
    )
    
    response = llm.invoke(messages + [write_prompt])
    
    return {"messages": [response], "next_agent": "end"}


# ===== 条件路由 =====

def should_continue(state: AgentState) -> str:
    """决定是否继续"""
    if state["next_agent"] == "end":
        return END
    return state["next_agent"]


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
    print(f"最后一条消息: {result['messages'][-1].content[:200]}...")
    print()


def demo_conditional_routing():
    """演示：条件路由"""
    print("=" * 50)
    print("2. 条件路由")
    print("=" * 50)
    
    # 定义条件节点
    def router_node(state: AgentState) -> AgentState:
        """路由节点：决定下一个 Agent"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        last_message = state["messages"][-1]
        
        # 判断问题类型
        prompt = HumanMessage(content=f"""
根据用户问题判断应该交给哪个 Agent 处理：
- 研究问题 → 返回 "researcher"
- 分析问题 → 返回 "analyst"  
- 其他 → 返回 "end"

用户问题：{last_message.content}

只返回一个词：researcher、analyst 或 end
""")
        
        response = llm.invoke([prompt])
        next_agent = response.content.strip().lower()
        
        if next_agent not in ["researcher", "analyst"]:
            next_agent = "end"
        
        return {"messages": [response], "next_agent": next_agent}
    
    # 创建图
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    
    # 设置入口
    workflow.set_entry_point("router")
    
    # 条件边
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_agent"],
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "end": END
        }
    )
    
    workflow.add_edge("researcher", END)
    workflow.add_edge("analyst", END)
    
    # 编译
    app = workflow.compile()
    
    # 测试
    result = app.invoke({
        "messages": [HumanMessage(content="帮我研究一下 Python")],
        "next_agent": ""
    })
    
    print(f"路由结果: {result['next_agent']}")
    print()


def demo_graph_visualization():
    """演示：图的描述"""
    print("=" * 50)
    print("3. LangGraph 架构图")
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


def demo_async():
    """演示：异步执行"""
    print("=" * 50)
    print("4. 异步执行")
    print("=" * 50)
    
    async def run_async():
        workflow = StateGraph(AgentState)
        
        workflow.add_node("researcher", researcher_node)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("writer", writer_node)
        
        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", END)
        
        app = workflow.compile()
        
        # 异步调用
        result = await app.ainvoke({
            "messages": [HumanMessage(content="Python 是什么？")],
            "next_agent": "researcher"
        })
        
        return result
    
    # 运行
    result = asyncio.run(run_async())
    print(f"异步执行完成，消息数: {len(result['messages'])}")
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：LangGraph（多 Agent 协作）")
    print("=" * 50 + "\n")
    
    # 简单工作流
    demo_simple_workflow()
    
    # 条件路由
    demo_conditional_routing()
    
    # 架构图
    demo_graph_visualization()
    
    # 异步
    demo_async()
    
    print("=" * 50)
    print("✅ 场景七演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
