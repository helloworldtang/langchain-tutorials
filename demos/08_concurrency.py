"""并发处理

演示内容：
1. asyncio 异步调用
2. RunnableParallel 并发
3. 批量处理

运行：uv run python demos/08_concurrency.py
"""
import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

load_dotenv()


def get_llm(temperature=0):
    """获取 DeepSeek LLM 实例"""
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature
    )


def demo_sync_vs_async():
    """演示：同步 vs 异步"""
    print("=" * 50)
    print("1. 同步 vs 异步对比")
    print("=" * 50)
    
    llm = get_llm()
    
    questions = [
        "什么是 Python？",
        "什么是 Java？",
    ]
    
    # 同步调用
    print("同步调用（串行）：")
    start = time.time()
    for q in questions:
        llm.invoke(q)
    sync_time = time.time() - start
    print(f"  耗时: {sync_time:.2f}s")
    
    # 异步调用
    print("\n异步调用（并发）：")
    start = time.time()
    
    async def async_call():
        tasks = [llm.ainvoke(q) for q in questions]
        return await asyncio.gather(*tasks)
    
    asyncio.run(async_call())
    async_time = time.time() - start
    print(f"  耗时: {async_time:.2f}s")
    
    print(f"\n💡 性能提升: {(sync_time - async_time) / sync_time * 100:.1f}%")
    print()


def demo_runnable_parallel():
    """演示：RunnableParallel"""
    print("=" * 50)
    print("2. RunnableParallel（同一输入，多任务并发）")
    print("=" * 50)
    
    llm = get_llm()
    
    summary_prompt = ChatPromptTemplate.from_template("用一句话总结：{text}")
    sentiment_prompt = ChatPromptTemplate.from_template("分析情感（正面/负面/中性）：{text}")
    keywords_prompt = ChatPromptTemplate.from_template("提取3个关键词：{text}")
    
    parallel_chain = RunnableParallel(
        summary=summary_prompt | llm,
        sentiment=sentiment_prompt | llm,
        keywords=keywords_prompt | llm,
    )
    
    text = "LangChain 是一个优秀的 LLM 应用框架，它让开发者能够快速构建 AI 应用。"
    
    start = time.time()
    result = parallel_chain.invoke({"text": text})
    elapsed = time.time() - start
    
    print(f"输入: {text}")
    print(f"\n摘要: {result['summary'].content}")
    print(f"情感: {result['sentiment'].content}")
    print(f"关键词: {result['keywords'].content}")
    print(f"\n耗时: {elapsed:.2f}s（3个任务并发执行）")
    print()


def demo_batch_processing():
    """演示：批量处理"""
    print("=" * 50)
    print("3. 批量处理")
    print("=" * 50)
    
    llm = get_llm()
    
    inputs = [
        "翻译成英文：你好",
        "翻译成英文：谢谢",
    ]
    
    print("批量调用：")
    start = time.time()
    results = llm.batch(inputs)
    batch_time = time.time() - start
    
    for i, r in enumerate(results):
        print(f"  {inputs[i]} → {r.content}")
    
    print(f"\n耗时: {batch_time:.2f}s")
    print()


def demo_with_semaphore():
    """演示：并发控制"""
    print("=" * 50)
    print("4. 并发控制（信号量）")
    print("=" * 50)
    
    llm = get_llm()
    
    async def limited_call(semaphore, prompt):
        async with semaphore:
            return await llm.ainvoke(prompt)
    
    async def process_with_limit():
        semaphore = asyncio.Semaphore(2)  # 限制 2 并发
        prompts = [f"数字{i}的平方是多少？" for i in range(4)]
        tasks = [limited_call(semaphore, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    start = time.time()
    asyncio.run(process_with_limit())
    elapsed = time.time() - start
    
    print(f"处理 4 个任务，限制 2 并发")
    print(f"耗时: {elapsed:.2f}s")
    print()
    print("💡 为什么要控制并发？")
    print("  1. API 有速率限制")
    print("  2. 避免服务器过载")
    print("  3. 成本控制")
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：并发处理")
    print("=" * 50 + "\n")
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 错误：请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    demo_sync_vs_async()
    demo_runnable_parallel()
    demo_batch_processing()
    demo_with_semaphore()
    
    print("""
并发场景选择：

| 场景 | 是否需要并发 | 原因 |
|------|-------------|------|
| 单次对话 | ❌ 不需要 | 只有一个任务 |
| 批量处理文档 | ✅ 必须 | 时间差 10x+ |
| 多维度分析 | ✅ 推荐 | 时间差 3-5x |
""")
    
    print("=" * 50)
    print("✅ 场景八演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
