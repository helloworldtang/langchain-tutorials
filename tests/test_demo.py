"""LangChain 教程测试"""
import unittest
import os


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
    
    def test_with_structured_output(self):
        """测试 with_structured_output 方法"""
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel
        
        class Answer(BaseModel):
            value: int
        
        # 只有配置了 API Key 才测试
        if os.environ.get("OPENAI_API_KEY"):
            llm = ChatOpenAI(model="gpt-4o-mini")
            structured_llm = llm.with_structured_output(Answer)
            self.assertIsNotNone(structured_llm)
        else:
            # 跳过测试
            self.skipTest("OPENAI_API_KEY not set")


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


if __name__ == "__main__":
    unittest.main()
