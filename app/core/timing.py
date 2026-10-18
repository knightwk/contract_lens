"""分段耗时打点。

为什么需要
    Day 6 的自检要求「拆出 embedding / 检索 / LLM 各段占比」。
    只有总耗时无法定位瓶颈 —— 慢 3 秒，是模型慢、检索慢、还是 LLM 慢？

为什么是显式对象，不是装饰器、也不是全局变量
    打点要跨 retrieve → rag.answer → LLM 三层，结果要一路带回 API 响应。

    · 全局变量：FastAPI 会【并发】处理多个请求，全局状态会互相串
    · 装饰器：只能装饰单个函数，没法把跨层的分段汇总出来

    所以显式传一个对象最不容易出错 —— 代价是每个函数多一个参数。

同名 span 会累加
    一次请求里可能有多次同类调用（比如分批编码），
    累加比覆盖更接近"这一段总共花了多久"。
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


class Timing:
    """一次请求内的分段计时器。

    用法：
        timing = Timing()
        with timing.span("search"):
            hits = await retrieve(...)
        return timing.to_dict()
    """

    def __init__(self) -> None:
        self._segments: dict[str, float] = {}
        self._t0 = perf_counter()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """给一段代码计时（毫秒）。异常也会记录耗时 —— 失败也要知道花了多久。"""
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = (perf_counter() - start) * 1000
            self._segments[name] = self._segments.get(name, 0.0) + elapsed

    def add(self, name: str, ms: float) -> None:
        """补记一段【在别处已经测好】的耗时。

        用途：LLM 的耗时由 llm.py 自己算（它要区分重试与真实生成），
        这里只把它并进来，不重复计时。
        """
        self._segments[name] = self._segments.get(name, 0.0) + ms

    def to_dict(self) -> dict:
        return {
            "total_ms": round((perf_counter() - self._t0) * 1000, 1),
            "segments": {k: round(v, 1) for k, v in self._segments.items()},
        }
