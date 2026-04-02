# LangChain 教程：从 0 到 1 玩转 LangChain

> 涵盖 LLM 调用、Function Call、Memory、RAG、结构化输出、Agent、LangGraph、并发处理

## 项目介绍

本项目是 **LangChain 入门教程的配套代码**，包含 8 个核心场景的完整示例，使用 DeepSeek 模型。

## 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/helloworldtang/langchain-tutorials.git
cd langchain-tutorials

# 2. 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装依赖
uv sync

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 5. 运行示例
uv run python demos/01_llm.py
uv run python demos/02_function_call.py
uv run python demos/03_memory.py
uv run python demos/04_rag.py
uv run python demos/05_structured_output.py
uv run python demos/06_agent.py
uv run python demos/07_langgraph.py
uv run python demos/08_concurrency.py

# 6. 运行测试
uv run pytest tests/ -v
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
├── pyproject.toml             # 项目配置
└── README.md
```

## 场景列表

| 场景 | 文件 | 内容 |
|------|------|------|
| 1️⃣ LLM 调用 | `demos/01_llm.py` | 基本调用、流式输出 |
| 2️⃣ Function Call | `demos/02_function_call.py` | 工具定义、绑定、自动执行 |
| 3️⃣ Memory | `demos/03_memory.py` | 对话记忆、Buffer/Summary |
| 4️⃣ RAG | `demos/04_rag.py` | 文档切分、向量数据库、检索生成 |
| 5️⃣ 结构化输出 | `demos/05_structured_output.py` | Pydantic、输出解析 |
| 6️⃣ Agent | `demos/06_agent.py` | 智能体、工具调用 |
| 7️⃣ LangGraph | `demos/07_langgraph.py` | 多 Agent 协作、状态图 |
| 8️⃣ 并发处理 | `demos/08_concurrency.py` | 异步、批量、并发控制 |

## 环境变量

```bash
# 必需
DEEPSEEK_API_KEY=sk-xxx

# 可选（RAG 场景需要 Embeddings）
OPENAI_API_KEY=sk-xxx
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

## 技术栈

- Python 3.10+
- LangChain 0.3+
- DeepSeek（国产大模型）
- FAISS（向量数据库）
- Pydantic（结构化输出）
- LangGraph（多 Agent）
- uv（依赖管理）

## 参考

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [DeepSeek API](https://platform.deepseek.com/)

## License

MIT
