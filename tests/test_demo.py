"""测试 Demo"""
import unittest

class TestDemo(unittest.TestCase):
    def test_langchain_imports(self):
        from langchain import ChatOpenAI, HumanMessage
        self.assertIsNotNone(ChatOpenAI)
    
    def test_tool(self):
        from langchain import Tool
        def add(a,b): return a+b
        t = Tool(name="add", func=add)
        self.assertEqual(t.run(a=1,b=2), 3)

if __name__ == "__main__":
    unittest.main()
