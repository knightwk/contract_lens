"""检索：把向量库的能力收口成"给问题、拿材料"的接口。

为什么单独成文件
    vector_store 只管 Milvus 的读写，不关心"为谁检索、检索多少、怎么排序"。
    W2 会在这里加混合检索与 Cross-Encoder 精排，W4 会在这里接检索评测 ——
    都需要一个稳定的收口。

Day 5 的范围
    最朴素的 Top-K。精排是 W2 Day 10 的事，混合检索的评估是 W2 Day 13 的事。
"""

from app.config import settings
from app.core.embedding import get_embedding_provider
from app.core.schemas import SearchHit
from app.core.timing import Timing
from app.core.vector_store import vector_store


async def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    expr: str | None = None,
    timing: Timing | None = None,
) -> list[SearchHit]:
    """按问题检索相关条款，返回带完整定位信息的材料。

    走【单向量】路径，这是有意的：
        W2 Day 13 要做三方案对比（A 单向量 / B 混合+RRF / C 混合+精排），
        A 是那份对照实验的基线。这里若直接用最好的那条，W2 就没有对照物了。
        （见 docs/decisions.md D-023）

    top_k 偏宽是「宽进严出」的前半段 —— 收紧交给 W2 的 Cross-Encoder 精排。

    Args:
        query: 用户问题原文
        top_k: 返回条数，缺省取 settings.rag_top_k
        expr: Milvus 标量过滤表达式（W3 的权限过滤会用到，现在传 None）
        timing: 分段耗时收集器。不传就建一个临时的（调用方不关心耗时的情况）。
    """
    top_k = top_k or settings.rag_top_k
    timing = timing or Timing()

    # 这两段是检索的耗时构成：编码问题和查库是两种完全不同的开销，
    # 分开打点才知道瓶颈在哪（模型慢？还是 Milvus 慢？）
    with timing.span("embed_query"):
        query_vec = await get_embedding_provider().embed_query(query)
    with timing.span("search_dense"):
        return await vector_store.search_dense(vector=query_vec, limit=top_k, expr=expr)
