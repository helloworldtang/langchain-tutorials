"""
场景八：并发处理

演示内容：
1. asyncio 异步调用
2. RunnableParallel 并发
3. 批量处理

运行：python demos/08_concurrency.py
"""
import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()


def demo_sync_vs_async():
    """演示：同步 vs 异步"""
    print("=" * 50)
    print("1. 同步 vs 异步对比")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    questions = [
        "什么是 Python？",
        "什么是 Java？",
        "什么是 Go？",
    ]
    
    # 同步调用
    print("同步调用（串行）：")
    start = time.time()
    results = []
    for q in questions:
        r = llm.invoke(q)
        results.append(r.content[:30])
    sync_time = time.time() - start
    print(f"  耗时: {sync_time:.2f}s")
    
    # 异步调用
    print("\n异步调用（并发）：")
    start = time.time()
    
    async def async_call():
        tasks = [llm.ainvoke(q) for q in questions]
        return await asyncio.gather(*tasks)
    
    results = asyncio.run(async_call())
    async_time = time.time() - start
    print(f"  耗时: {async_time:.2f}s")
    
    print(f"\n💡 性能提升: {(sync_time - async_time) / sync_time * 100:.1f}%")
    print()


def demo_runnable_parallel():
    """演示：RunnableParallel"""
    print("=" * 50)
    print("2. RunnableParallel（同一输入，多任务并发）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 定义多个 Prompt
    summary_prompt = ChatPromptTemplate.from_template("用一句话总结：{text}")
    sentiment_prompt = ChatPromptTemplate.from_template("分析情感（正面/负面/中性）：{text}")
    keywords_prompt = ChatPromptTemplate.from_template("提取3个关键词：{text}")
    
    # 创建并行链
    parallel_chain = RunnableParallel(
        summary=summary_prompt | llm,
        sentiment=sentiment_prompt | llm,
        keywords=keywords_prompt | llm,
    )
    
    # 执行
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
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 批量输入
    inputs = [
        "翻译成英文：你好",
        "翻译成英文：谢谢",
        "翻译成英文：再见",
    ]
    
    # 方式1：批量调用（自动批处理）
    print("方式1：批量调用")
    start = time.time()
    results = llm.batch(inputs)
    batch_time = time.time() - start
    print(f"  耗时: {batch_time:.2f}s")
    
    # 方式2：串行调用
    print("\n方式2：串行调用")
    start = time.time()
    results_serial = [llm.invoke(i) for i in inputs]
    serial_time = time.time() - start
    print(f"  耗时: {serial_time:.2f}s")
    
    print(f"\n💡 批量处理更快，减少 API 往返")
    print()


def demo_async_batch():
    """演示：异步批量处理"""
    print("=" * 50)
    print("4. 异步批量处理（推荐用于大规模）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    async def process_batch(items, batch_size=5):
        """分批异步处理"""
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            tasks = [llm.ainvoke(item) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"  已处理 {min(i+batch_size, len(items))}/{len(items)}")
        return results
    
    # 模拟大量任务
    items = [f"用一句话解释：什么是{i+1}？" for i in range(10)]
    
    print(f"处理 {len(items)} 个任务...")
    start = time.time()
    results = asyncio.run(process_batch(items, batch_size=3))
    elapsed = time.time() - start
    
    print(f"\n总耗时: {elapsed:.2f}s")
    print(f"平均每个: {elapsed/len(items)*1000:.0f}ms")
    print()


def demo_with_semaphore():
    """演示：并发控制"""
    print("=" * 50)
    print("5. 并发控制（信号量）")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    async def limited_call(semaphore, prompt):
        """受限并发调用"""
        async with semaphore:
            return await llm.ainvoke(prompt)
    
    async def process_with_limit():
        # 限制最多 3 个并发
        semaphore = asyncio.Semaphore(3)
        
        prompts = [f"数字{i}的平方是多少？" for i in range(10)]
        
        tasks = [limited_call(semaphore, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    start = time.time()
    results = asyncio.run(process_with_limit())
    elapsed = time.time() - start
    
    print(f"处理 10 个任务，限制 3 并发")
    print(f"耗时: {elapsed:.2f}s")
    print()
    print("💡 为什么要控制并发？")
    print("  1. API 有速率限制（如 OpenAI 60次/分钟）")
    print("  2. 避免服务器过载")
    print("  3. 成本控制")
    print()


def demo_timeout():
    """演示：超时控制"""
    print("=" * 50)
    print("6. 超时控制")
    print("=" * 50)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=10)
    
    async def call_with_timeout(prompt, timeout=5):
        """带超时的调用"""
        try:
            result = await asyncio.wait_for(
                llm.ainvoke(prompt),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return "请求超时"
    
    result = asyncio.run(call_with_timeout("你好", timeout=10))
    print(f"结果: {result.content if hasattr(result, 'content') else result}")
    print()
    print("💡 生产环境必须设置超时，避免任务卡死")
    print()


def main():
    print("\n" + "=" * 50)
    print("LangChain 入门：并发处理")
    print("=" * 50 + "\n")
    
    # 同步 vs 异步
    demo_sync_vs_async()
    
    # RunnableParallel
    demo_runnable_parallel()
    
    # 批量处理
    demo_batch_processing()
    
    # 异步批量
    demo_async_batch()
    
    # 并发控制
    demo_with_semaphore()
    
    # 超时
    demo_timeout()
    
    print("=" * 50)
    print("✅ 场景八演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
