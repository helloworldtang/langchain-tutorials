"""结构化输出（Structured Output）

演示内容：
1. Pydantic 模型定义
2. PydanticOutputParser 解析
3. 结构化数据提取

运行：uv run python demos/05_structured_output.py
"""
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()


def get_llm(temperature=0):
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature
    )


# ===== 定义数据模型 =====

class PersonInfo(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
    hobbies: List[str] = Field(description="爱好列表")


class ProductInfo(BaseModel):
    """商品信息"""
    name: str = Field(description="商品名称")
    price: float = Field(description="价格")
    category: str = Field(description="分类")
    features: List[str] = Field(description="特点列表")


class StockAnalysis(BaseModel):
    """股票分析结果"""
    symbol: str = Field(description="股票代码")
    company_name: str = Field(description="公司名称")
    current_price: float = Field(description="当前价格")
    trend: str = Field(description="趋势：上涨/下跌/横盘")
    recommendation: str = Field(description="建议：买入/持有/卖出")
    reasons: List[str] = Field(description="分析理由")


# ===== 演示函数 =====

def demo_pydantic_parser():
    """演示：PydanticOutputParser"""
    print("=" * 50)
    print("1. PydanticOutputParser（结构化输出）")
    print("=" * 50)
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。{format_instructions}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | parser
    
    query = "请告诉我关于马斯克的信息"
    result = chain.invoke({
        "query": query,
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"查询: {query}")
    print(f"结果类型: {type(result)}")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"职业: {result.occupation}")
    print(f"爱好: {result.hobbies}")
    print()


def demo_product_info():
    """演示：商品信息提取"""
    print("=" * 50)
    print("2. 商品信息提取")
    print("=" * 50)
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=ProductInfo)
    
    prompt = ChatPromptTemplate.from_template("""
{format_instructions}

请分析以下产品：{product}
""")
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "product": "iPhone 15",
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"商品名: {result.name}")
    print(f"价格: {result.price}")
    print(f"分类: {result.category}")
    print(f"特点: {result.features}")
    print()


def demo_stock_analysis():
    """演示：股票分析"""
    print("=" * 50)
    print("3. 股票分析（复杂数据结构）")
    print("=" * 50)
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=StockAnalysis)
    
    prompt = ChatPromptTemplate.from_template("""
你是一位专业的股票分析师。

{format_instructions}

请分析以下股票：{symbol}
""")
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "symbol": "茅台（600519）",
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"股票代码: {result.symbol}")
    print(f"公司名称: {result.company_name}")
    print(f"当前价格: {result.current_price}")
    print(f"趋势: {result.trend}")
    print(f"建议: {result.recommendation}")
    print(f"理由: {result.reasons}")
    print()


def demo_list_output():
    """演示：列表输出"""
    print("=" * 50)
    print("4. 列表输出")
    print("=" * 50)
    
    class TodoList(BaseModel):
        """待办事项列表"""
        date: str = Field(description="日期")
        items: List[str] = Field(description="待办事项")
        priority: str = Field(description="优先级描述")
    
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=TodoList)
    
    prompt = ChatPromptTemplate.from_template("""
{format_instructions}

帮我列一个程序员明天的待办事项
""")
    
    chain = prompt | llm | parser
    
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"日期: {result.date}")
    print(f"优先级: {result.priority}")
    print("待办事项:")
    for i, item in enumerate(result.items, 1):
        print(f"  {i}. {item}")
    print()


def demo_json_output():
    """演示：JSON 格式说明"""
    print("=" * 50)
    print("5. 格式说明示例")
    print("=" * 50)
    
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    print("PydanticOutputParser 会生成以下格式说明：\n")
    print(parser.get_format_instructions())
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：结构化输出")
    print("=" * 50 + "\n")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    demo_pydantic_parser()
    demo_product_info()
    demo_stock_analysis()
    demo_list_output()
    
    print("""
Pydantic 模型定义要点：

┌─────────────────────────────────────────────────────────────┐
│  class ProductInfo(BaseModel):                              │
│      name: str = Field(description="商品名称")              │
│      price: float = Field(description="价格")               │
│      features: List[str] = Field(description="特点列表")    │
└─────────────────────────────────────────────────────────────┘

💡 关键点：
1. 继承 BaseModel
2. 使用 Field 添加描述（LLM 会根据描述生成内容）
3. 支持基本类型：str、int、float、bool
4. 支持复杂类型：List、Optional、嵌套模型
""")
    
    print("=" * 50)
    print("✅ 场景五演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
