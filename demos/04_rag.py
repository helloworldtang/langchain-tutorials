"""RAG（检索增强生成）

演示内容：
1. 文档切分
2. 向量化存储（FAISS）
3. 检索 + 生成回答

运行：uv run python demos/04_rag.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from helpers import get_llm, get_embeddings

load_dotenv()


# 示例文档
SAMPLE_DOCUMENTS = [
    """
    Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年发布。
    Python 的设计哲学强调代码的可读性和简洁性。
    Python 支持多种编程范式，包括面向对象、函数式和过程式编程。
    """,
    """
    LangChain 是一个用于开发大语言模型应用的框架。
    它提供了链式调用、Agent、Memory、RAG 等核心组件。
    LangChain 支持多种 LLM 后端，包括 DeepSeek、OpenAI 等。
    """,
    """
    RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。
    它首先从知识库中检索相关文档，然后将文档作为上下文传递给 LLM。
    RAG 可以让 LLM 回答它训练数据中没有的问题。
    """,
    """
    FAISS 是 Facebook 开发的向量相似度搜索库。
    它支持高维向量的高效检索，适合处理大规模向量数据。
    LangChain 内置了 FAISS 的集成，可以轻松构建向量数据库。
    """,
]


def demo_text_splitting() -> None:
    """演示：文档切分"""
    print("=" * 50)
    print("1. 文档切分")
    print("=" * 50)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " "]
    )

    text = SAMPLE_DOCUMENTS[0]
    chunks = text_splitter.split_text(text)

    print(f"原文长度: {len(text)} 字符")
    print(f"切分后: {len(chunks)} 块\n")

    for i, chunk in enumerate(chunks):
        print(f"块 {i+1}: {chunk[:50]}...")
    print()


def demo_rag_pipeline() -> None:
    """演示：完整的 RAG 流程"""
    print("=" * 50)
    print("2. 完整 RAG 流程")
    print("=" * 50)

    # 检查 Embeddings API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("注意：RAG 需要 Embeddings API")
        print("   请设置 OPENAI_API_KEY 环境变量")
        print("   或者使用本地 Embeddings（如 sentence-transformers）")
        return

    # 1. 准备数据
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    documents = []
    for doc in SAMPLE_DOCUMENTS:
        chunks = text_splitter.split_text(doc)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))

    # 2. 创建向量数据库
    print(f"创建向量数据库，共 {len(documents)} 个文档块...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 3. 创建 LLM
    llm = get_llm()

    # 4. 创建 RAG Prompt
    prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答：
""")

    # 5. 构建 RAG 链
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # 6. 测试
    questions = [
        "什么是 RAG？",
        "LangChain 有哪些核心组件？",
    ]

    for question in questions:
        print(f"问题: {question}")
        response = rag_chain.invoke(question)
        print(f"回答: {response.content}\n")


def demo_rag_with_sources() -> None:
    """演示：带来源引用的 RAG"""
    print("=" * 50)
    print("3. 带来源引用的 RAG")
    print("=" * 50)

    if not os.getenv("OPENAI_API_KEY"):
        print("需要 OPENAI_API_KEY 来生成 Embeddings")
        return

    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    documents = []
    for i, doc in enumerate(SAMPLE_DOCUMENTS):
        chunks = text_splitter.split_text(doc)
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={"source": f"文档{i+1}"}
            ))

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    query = "RAG 有什么作用？"
    docs = retriever.invoke(query)

    print(f"问题: {query}")
    print(f"\n找到 {len(docs)} 个相关文档:\n")

    for i, doc in enumerate(docs):
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"内容: {doc.page_content[:100]}...")
        print()


def main() -> None:
    print("\n" + "=" * 50)
    print("LangChain 入门：RAG（检索增强生成）")
    print("=" * 50 + "\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return

    demo_text_splitting()
    demo_rag_pipeline()
    demo_rag_with_sources()

    print("=" * 50)
    print("场景四演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
