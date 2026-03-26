# LangChain 教程：从 0 到 1 玩转 LangChain

> 涵盖 LLM 调用、Function Call、MCP、Skill、Agent、Multi-Agent、多线程

## 项目介绍

本项目展示 LangChain 核心概念，使用 DeepSeek 模型，包含完整代码和测试。

## 目录结构

```
langchain-tutorials/
├── langchain/           # 简化版 LangChain 框架
│   └── __init__.py
├── demos/              # 示例代码
│   ├── 01_llm.py           # LLM 直接调用
│   ├── 02_function_call.py # Function Call
│   ├── 03_mcp.py           # MCP 协议
│   ├── 04_skill.py         # Skill 技能系统
│   ├── 05_agent.py          # Agent 智能体
│   ├── 06_multi_agent.py    # Multi-Agent
│   └── 07_threading.py     # 多线程并发
├── tests/              # 单元测试
│   └── test_demo.py
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 克隆项目
git clone https://github.com/helloworldtang/langchain-tutorials.git
cd langchain-tutorials

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export DEEPSEEK_API_KEY="sk-xxx"

# 运行示例
python demos/01_llm.py
python demos/02_function_call.py
python demos/03_mcp.py
python demos/04_skill.py
python demos/05_agent.py
python demos/06_multi_agent.py
python demos/07_threading.py

# 运行测试
pytest tests/ -v
```

## 概念讲解

### 1. LLM 直接调用

```python
from langchain import ChatOpenAI, HumanMessage

llm = ChatOpenAI(model="deepseek-chat")
response = llm.invoke([HumanMessage("你好")])
print(response.content)
```

### 2. Function Call

让 LLM 调用外部工具（天气、计算等）

### 3. MCP

Model Context Protocol，标准化工具连接协议

### 4. Skill

封装好的能力模块，可复用

### 5. Agent

能自主规划执行的智能体

### 6. Multi-Agent

多个 Agent 协作完成复杂任务

### 7. Threading

多线程并发处理，提高效率

## 测试

```bash
$ pytest tests/ -v

tests/test_demo.py::TestDemo::test_langchain_imports PASSED
tests/test_demo.py::TestDemo::test_tool PASSED

======================== 2 passed in 0.09s ========================
```

## 技术栈

- Python 3.10+
- OpenAI SDK
- DeepSeek 模型

## 参考

- [DeepSeek API](https://platform.deepseek.com/)
- [LangChain](https://www.langchain.com/)

## License

MIT
