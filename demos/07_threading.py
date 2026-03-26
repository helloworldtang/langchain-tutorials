"""多线程并发"""
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as e: print(list(e.map(lambda x: x*2, [1,2,3])))
