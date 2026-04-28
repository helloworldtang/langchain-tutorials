"""LangChain 教程测试"""
import unittest
import os
import json


class TestLLM(unittest.TestCase):
    """测试 LLM 调用"""

    def test_imports(self):
        """测试核心导入"""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from langchain_core.prompts import ChatPromptTemplate

        self.assertIsNotNone(ChatOpenAI)
        self.assertIsNotNone(HumanMessage)
        self.assertIsNotNone(AIMessage)
        self.assertIsNotNone(SystemMessage)
        self.assertIsNotNone(ChatPromptTemplate)

    def test_message_creation(self):
        """测试消息创建"""
        from langchain_core.messages import HumanMessage, AIMessage

        human_msg = HumanMessage(content="你好")
        ai_msg = AIMessage(content="你好！有什么可以帮助你的？")

        self.assertEqual(human_msg.content, "你好")
        self.assertEqual(ai_msg.content, "你好！有什么可以帮助你的？")

    def test_prompt_template(self):
        """测试 Prompt 模板"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template("你好，{name}！")
        result = prompt.invoke({"name": "小明"})

        self.assertIn("小明", result.to_string())


class TestTools(unittest.TestCase):
    """测试工具"""

    def test_tool_decorator(self):
        """测试 @tool 装饰器"""
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            """加法运算"""
            return a + b

        self.assertEqual(add.name, "add")
        self.assertIn("加法", add.description)

    def test_tool_invoke(self):
        """测试工具调用"""
        from langchain_core.tools import tool

        @tool
        def multiply(a: int, b: int) -> int:
            """乘法运算"""
            return a * b

        result = multiply.invoke({"a": 3, "b": 4})
        self.assertEqual(result, 12)


class TestMemory(unittest.TestCase):
    """测试记忆"""

    def test_in_memory_history(self):
        """测试 InMemoryChatMessageHistory"""
        from langchain_core.chat_history import InMemoryChatMessageHistory

        history = InMemoryChatMessageHistory()
        history.add_user_message("你好")
        history.add_ai_message("你好！")

        self.assertEqual(len(history.messages), 2)

    def test_chat_message_history(self):
        """测试 ChatMessageHistory"""
        from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

        history = ChatMessageHistory()
        history.add_user_message("你好")
        history.add_ai_message("你好！")

        self.assertEqual(len(history.messages), 2)


class TestStructuredOutput(unittest.TestCase):
    """测试结构化输出"""

    def test_pydantic_model(self):
        """测试 Pydantic 模型"""
        from pydantic import BaseModel, Field

        class Person(BaseModel):
            name: str = Field(description="姓名")
            age: int = Field(description="年龄")

        person = Person(name="小明", age=25)

        self.assertEqual(person.name, "小明")
        self.assertEqual(person.age, 25)


class TestRAG(unittest.TestCase):
    """测试 RAG"""

    def test_text_splitter(self):
        """测试文本切分"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = "这是第一句话。这是第二句话。这是第三句话。"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=20,
            chunk_overlap=5
        )

        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 0)

    def test_document_creation(self):
        """测试文档创建"""
        from langchain_core.documents import Document

        doc = Document(
            page_content="测试内容",
            metadata={"source": "test"}
        )

        self.assertEqual(doc.page_content, "测试内容")
        self.assertEqual(doc.metadata["source"], "test")


class TestConcurrency(unittest.TestCase):
    """测试并发"""

    def test_runnable_parallel_import(self):
        """测试 RunnableParallel 导入"""
        from langchain_core.runnables import RunnableParallel

        self.assertIsNotNone(RunnableParallel)

    def test_asyncio_gather(self):
        """测试 asyncio.gather"""
        import asyncio

        async def async_task(n):
            return n * 2

        async def run():
            results = await asyncio.gather(
                async_task(1),
                async_task(2),
                async_task(3)
            )
            return results

        results = asyncio.run(run())
        self.assertEqual(results, [2, 4, 6])


class TestCommonModule(unittest.TestCase):
    """测试公共模块"""

    def test_get_llm_returns_chat_openai(self):
        """测试 get_llm 返回正确类型"""
        from helpers import get_llm
        from langchain_openai import ChatOpenAI

        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        llm = get_llm()
        self.assertIsInstance(llm, ChatOpenAI)
        del os.environ["DEEPSEEK_API_KEY"]

    def test_get_llm_temperature(self):
        """测试 get_llm temperature 参数"""
        from helpers import get_llm

        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        llm = get_llm(temperature=0.5)
        self.assertEqual(llm.temperature, 0.5)

        llm_default = get_llm()
        self.assertEqual(llm_default.temperature, 0)
        del os.environ["DEEPSEEK_API_KEY"]

    def test_get_embeddings_returns_correct_type(self):
        """测试 get_embeddings 返回正确类型"""
        from helpers import get_embeddings
        from langchain_openai import OpenAIEmbeddings

        os.environ["OPENAI_API_KEY"] = "test-key"
        embeddings = get_embeddings()
        self.assertIsInstance(embeddings, OpenAIEmbeddings)
        del os.environ["OPENAI_API_KEY"]

    def test_deepseek_api_base_default(self):
        """测试 DEEPSEEK_API_BASE 默认值"""
        from helpers import DEEPSEEK_API_BASE

        # 默认值
        env_val = os.environ.pop("DEEPSEEK_API_BASE", None)
        # 需要重新导入获取默认值
        import importlib
        import helpers
        importlib.reload(helpers)
        self.assertEqual(helpers.DEEPSEEK_API_BASE, "https://api.deepseek.com/v1")
        if env_val:
            os.environ["DEEPSEEK_API_BASE"] = env_val


class TestSafeCalculate(unittest.TestCase):
    """测试安全计算函数"""

    def test_basic_addition(self):
        """测试基本加法"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate("2+3"), "计算结果: 5")

    def test_multiplication(self):
        """测试乘法"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate("3*4"), "计算结果: 12")

    def test_complex_expression(self):
        """测试复杂表达式"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate("2+3*4"), "计算结果: 14")

    def test_float_calculation(self):
        """测试浮点数计算"""
        from helpers import safe_calculate
        result = safe_calculate("1.5*2")
        self.assertEqual(result, "计算结果: 3.0")

    def test_division(self):
        """测试除法"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate("10/2"), "计算结果: 5.0")

    def test_parentheses(self):
        """测试括号"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate("(2+3)*4"), "计算结果: 20")

    def test_reject_code_injection(self):
        """测试拒绝代码注入"""
        from helpers import safe_calculate
        result = safe_calculate("__import__('os').system('ls')")
        self.assertIn("非法字符", result)

    def test_reject_function_call(self):
        """测试拒绝函数调用"""
        from helpers import safe_calculate
        result = safe_calculate("print('hello')")
        self.assertIn("非法字符", result)

    def test_reject_power_operator(self):
        """测试拒绝幂运算"""
        from helpers import safe_calculate
        result = safe_calculate("2**10")
        self.assertIn("不支持", result)

    def test_reject_variable_access(self):
        """测试拒绝变量访问"""
        from helpers import safe_calculate
        result = safe_calculate("open('/etc/passwd').read()")
        self.assertIn("非法字符", result)

    def test_division_by_zero(self):
        """测试除零错误"""
        from helpers import safe_calculate
        result = safe_calculate("1/0")
        self.assertIn("计算错误", result)

    def test_whitespace_in_expression(self):
        """测试表达式中的空格"""
        from helpers import safe_calculate
        self.assertEqual(safe_calculate(" 2 + 3 "), "计算结果: 5")


class TestToolSecurity(unittest.TestCase):
    """测试工具安全性"""

    def test_no_eval_in_demos(self):
        """验证 demo 文件中不包含直接 eval 调用（公共模块除外）"""
        import re
        demo_dir = os.path.join(os.path.dirname(__file__), "..", "demos")
        for filename in os.listdir(demo_dir):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(demo_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            # 不允许 demo 中直接出现 eval( 调用
            self.assertNotRegex(
                content, r'(?<!\w)eval\s*\(',
                f"{filename} contains unsafe eval() call"
            )

    def test_no_bare_except_in_demos(self):
        """验证 demo 文件中不包含裸 except"""
        demo_dir = os.path.join(os.path.dirname(__file__), "..", "demos")
        for filename in os.listdir(demo_dir):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(demo_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()
            self.assertNotRegex(
                content, r'except\s*:',
                f"{filename} contains bare except clause"
            )


if __name__ == "__main__":
    unittest.main()
