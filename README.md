# LangChain 教程：从 0 到 1 玩转 LangChain

> 涵盖 LLM 调用、Function Call、Memory、RAG、结构化输出、Agent、LangGraph、并发处理

## 项目介绍

本项目是 **LangChain 入门教程的配套代码**，包含 8 个核心场景的完整示例：

| 场景 | 文件 | 内容 |
|------|------|------|
| 1️⃣ LLM 调用 | `demos/01_llm.py` | 基本调用、流式输出、国产模型 |
| 2️⃣ Function Call | `demos/02_function_call.py` | 工具定义、绑定、自动执行 |
| 3️⃣ Memory | `demos/03_memory.py` | 对话记忆、Buffer/Summary |
| 4️⃣ RAG | `demos/04_rag.py` | 文档切分、向量数据库、检索生成 |
| 5️⃣ 结构化输出 | `demos/05_structured_output.py` | Pydantic、输出解析 |
| 6️⃣ Agent | `demos/06_agent.py` | 智能体、工具调用 |
| 7️⃣ LangGraph | `demos/07_langgraph.py` | 多 Agent 协作、状态图 |
| 8️⃣ 并发处理 | `demos/08_concurrency.py` | 异步、批量、并发控制 |

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/helloworldtang/langchain-tutorials.git
cd langchain-tutorials

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY

# 4. 运行示例
python demos/01_llm.py
python demos/02_function_call.py
python demos/03_memory.py
python demos/04_rag.py
python demos/05_structured_output.py
python demos/06_agent.py
python demos/07_langgraph.py
python demos/08_concurrency.py

# 5. 运行测试
pytest tests/ -v
```

## 目录结构

```
langchain-tutorials/
├── demos/                      # 示例代码
│   ├── 01_llm.py                  # LLM 直接调用
│   ├── 02_function_call.py        # Function Call
│   ├── 03_memory.py               # 对话记忆
│   ├── 04_rag.py                  # RAG 检索增强
│   ├── 05_structured_output.py    # 结构化输出
│   ├── 06_agent.py                # Agent 智能体
│   ├── 07_langgraph.py            # LangGraph 多 Agent
│   └── 08_concurrency.py          # 并发处理
├── tests/                      # 单元测试
│   └── test_demo.py
├── .env.example               # 环境变量示例
├── requirements.txt           # 依赖
└── README.md
```

## 核心概念速览

### 1. LLM 调用

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="你好")])
print(response.content)
```

### 2. Function Call

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}：晴天，15°C"

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_weather])
```

### 3. Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="你好，我叫小明")
conversation.predict(input="我叫什么名字？")  # AI 会记住
```

### 4. RAG

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["文档内容..."], embeddings)

# 检索
docs = vectorstore.similarity_search("查询问题", k=2)
```

### 5. 结构化输出

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(description="商品名称")
    price: float = Field(description="价格")

structured_llm = llm.with_structured_output(Product)
result = structured_llm.invoke("iPhone 15")
print(result.name, result.price)
```

### 6. Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "北京天气怎么样？"})
```

### 7. LangGraph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="问题")]})
```

### 8. 并发处理

```python
import asyncio

# 异步并发
tasks = [llm.ainvoke(q) for q in questions]
results = await asyncio.gather(*tasks)

# RunnableParallel
from langchain_core.runnables import RunnableParallel

parallel_chain = RunnableParallel(
    summary=summary_prompt | llm,
    sentiment=sentiment_prompt | llm,
)
```

## 学习路线

```
入门（1-2 天）
├── 01_llm.py          # 学会基本调用
├── 02_function_call.py # 学会工具调用
└── 03_memory.py       # 学会对话记忆

进阶（3-5 天）
├── 04_rag.py              # 学会知识库检索
├── 05_structured_output.py # 学会结构化输出
└── 06_agent.py            # 学会智能体

高级（1 周+）
├── 07_langgraph.py   # 学会多 Agent 协作
└── 08_concurrency.py # 学会并发处理
```

## 常见问题

### Q: 如何使用 DeepSeek 等国产模型？

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key="your-deepseek-key"
)
```

### Q: 如何控制 token 消耗？

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1000  # 限制输出长度
)
```

### Q: 如何处理 API 超时？

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=30,
    max_retries=2
)
```

## 测试

```bash
$ pytest tests/ -v

tests/test_demo.py::TestLLM::test_imports PASSED
tests/test_demo.py::TestLLM::test_message_creation PASSED
tests/test_demo.py::TestLLM::test_prompt_template PASSED
tests/test_demo.py::TestTools::test_tool_decorator PASSED
tests/test_demo.py::TestTools::test_tool_invoke PASSED
tests/test_demo.py::TestMemory::test_buffer_memory PASSED
tests/test_demo.py::TestMemory::test_in_memory_history PASSED
tests/test_demo.py::TestStructuredOutput::test_pydantic_model PASSED
tests/test_demo.py::TestStructuredOutput::test_pydantic_parser PASSED
tests/test_demo.py::TestRAG::test_text_splitter PASSED
tests/test_demo.py::TestRAG::test_document_creation PASSED
tests/test_demo.py::TestConcurrency::test_runnable_parallel_import PASSED
tests/test_demo.py::TestConcurrency::test_asyncio_gather PASSED

======================== 13 passed in 1.23s ========================
```

## 技术栈

- Python 3.10+
- LangChain 0.3+
- OpenAI / DeepSeek
- FAISS（向量数据库）
- Pydantic（结构化输出）
- LangGraph（多 Agent）

## 参考

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [OpenAI API](https://platform.openai.com/)

## License

MIT
