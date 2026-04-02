"""
场景五：结构化输出（Structured Output）

演示内容：
1. Pydantic 模型定义
2. PydanticOutputParser 解析
3. with_structured_output（推荐方式）

运行：python demos/05_structured_output.py
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()


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
    rating: Optional[float] = Field(default=None, description="评分（可选）")


class StockAnalysis(BaseModel):
    """股票分析结果"""
    symbol: str = Field(description="股票代码")
    company_name: str = Field(description="公司名称")
    current_price: float = Field(description="当前价格")
    trend: str = Field(description="趋势：上涨/下跌/横盘")
    recommendation: str = Field(description="建议：买入/持有/卖出")
    reasons: List[str] = Field(description="分析理由")
    risk_level: str = Field(description="风险等级：低/中/高")


# ===== 演示函数 =====

def demo_pydantic_parser():
    """演示：PydanticOutputParser"""
    print("=" * 50)
    print("1. PydanticOutputParser")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 创建解析器
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    # 创建 Prompt（包含格式说明）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手。{format_instructions}"),
        ("human", "{query}")
    ])
    
    # 构建链
    chain = prompt | llm | parser
    
    # 执行
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


def demo_structured_output():
    """演示：with_structured_output（推荐）"""
    print("=" * 50)
    print("2. with_structured_output（推荐方式）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 直接绑定结构化输出
    structured_llm = llm.with_structured_output(ProductInfo)
    
    # 调用
    query = "请分析 iPhone 15 这款产品"
    result = structured_llm.invoke(query)
    
    print(f"查询: {query}")
    print(f"结果类型: {type(result)}")
    print(f"商品名: {result.name}")
    print(f"价格: {result.price}")
    print(f"分类: {result.category}")
    print(f"特点: {result.features}")
    print()


def demo_structured_output_with_prompt():
    """演示：带 Prompt 的结构化输出"""
    print("=" * 50)
    print("3. 带 Prompt 的结构化输出")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 创建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的股票分析师。请分析用户询问的股票。"),
        ("human", "{symbol}")
    ])
    
    # 构建链
    structured_llm = llm.with_structured_output(StockAnalysis)
    chain = prompt | structured_llm
    
    # 执行
    result = chain.invoke({"symbol": "茅台（600519）"})
    
    print(f"股票代码: {result.symbol}")
    print(f"公司名称: {result.company_name}")
    print(f"当前价格: {result.current_price}")
    print(f"趋势: {result.trend}")
    print(f"建议: {result.recommendation}")
    print(f"理由: {result.reasons}")
    print(f"风险等级: {result.risk_level}")
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
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(TodoList)
    
    result = structured_llm.invoke("帮我列一个程序员明天的待办事项")
    
    print(f"日期: {result.date}")
    print(f"优先级: {result.priority}")
    print("待办事项:")
    for i, item in enumerate(result.items, 1):
        print(f"  {i}. {item}")
    print()


def demo_error_handling():
    """演示：错误处理"""
    print("=" * 50)
    print("5. 错误处理")
    print("=" * 50)
    
    from langchain_core.exceptions import OutputParserException
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    # 模拟一个可能失败的解析
    try:
        # 如果 LLM 返回的格式不对，会抛出异常
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个有用的助手。{format_instructions}"),
            ("human", "{query}")
        ])
        
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "query": "今天天气怎么样？",  # 这个问题和 PersonInfo 无关
            "format_instructions": parser.get_format_instructions()
        })
        
        print(f"解析成功: {result}")
        
    except OutputParserException as e:
        print(f"解析失败: {e}")
        print("💡 提示：确保 Prompt 清晰指明需要的信息类型")
    print()


def demo_comparison():
    """对比两种方式"""
    print("=" * 50)
    print("6. 两种方式对比")
    print("=" * 50)
    
    print("""
┌─────────────────────┬─────────────────────┬─────────────────────┐
│ 方式                │ 优点                │ 缺点                │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ PydanticOutputParser│ 兼容性好，可自定义  │ 需要手动处理 Prompt │
│ with_structured_    │ 自动处理，代码简洁  │ 依赖模型支持        │
│ output              │                     │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘

💡 推荐：优先使用 with_structured_output
   - OpenAI、DeepSeek 等主流模型都支持
   - 代码更简洁，不易出错
""")


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：结构化输出")
    print("=" * 50 + "\n")
    
    # PydanticOutputParser
    demo_pydantic_parser()
    
    # with_structured_output
    demo_structured_output()
    
    # 带 Prompt
    demo_structured_output_with_prompt()
    
    # 列表输出
    demo_list_output()
    
    # 错误处理
    demo_error_handling()
    
    # 对比
    demo_comparison()
    
    print("=" * 50)
    print("✅ 场景五演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
