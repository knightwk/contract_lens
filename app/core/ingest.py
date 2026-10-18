"""入库编排：解析 -> 切分 -> 向量化 -> 写 Milvus。

为什么单独成文件
    vector_store 只管 Milvus，chunker 只管切分，embedding 只管编码。
    「把它们串起来」是第三种职责。
    W3 的幂等键、W4 的模型对比都会长在这里，所以值得有自己的位置。
"""

import asyncio

from app.core.chunker import chunk_document
from app.core.embedding import get_embedding_provider
from app.core.logger import logger
from app.core.parser import parse_document
from app.core.schemas import ParsedDocument
from app.core.vector_store import vector_store

# ── 关于"不要在 async 里跑阻塞调用"（Day 6 完成） ──
#
# ① 怎么找
#    看每个 async def 里有没有【不带 await】的同步调用。
#    Day 6 全仓找出三处，都已改为 asyncio.to_thread：
#      · 本文件的 chunk_document
#      · 本文件 index_path 的 parse_document
#      · app/api/v1/documents.py 上传接口的 parse_document
#
# ② 多慢才值得丢线程池
#    to_thread 本身有开销（约几十微秒），太快的函数丢进去反而更慢。
#    实测：parse_document 解 50 页 PDF 约 1500 ms —— 远超开销，必须丢；
#    chunk_document 只要几毫秒 —— 丢不丢都行，为了与仓库其它调用一致而丢了。
#
# ③ 丢线程池解决了什么、没解决什么
#    ✅ 解决：事件循环不再被卡住，别的请求能插进来。
#       实测（scripts/blocking_probe.py）：重活跑 1.5 秒期间，
#       轻任务的调度延迟从 1519 ms 降到 0.1 ms —— 约 13000 倍。
#    ❌ 没解决：这个函数本身没变快（1477 vs 1519 ms，差 3%）。
#       GIL 让两个线程的 CPU 工作仍然串行，并发上传不会让总 CPU 时间减半。
#    推论：想要真正的 CPU 并行要用【进程】（多 worker），不是线程。
#          to_thread 只保住了响应性，没给算力。


async def index_document(doc: ParsedDocument, *, strategy: str = "clause") -> dict:
    """把一份已解析的文档写入向量库。

    注意 embedding 走的是 get_embedding_provider()，不是直接调 model_runtime ——
    这样切到 API 实现时这里一行都不用改（编码约定 6 的直接收益）。
    """
    chunks = await asyncio.to_thread(chunk_document, doc, strategy=strategy)
    if not chunks:
        return {"doc_id": doc.doc_id, "chunks": 0, "written": 0}

    provider = get_embedding_provider()
    # 批量编码：CPU 上一次编码太多会吃满内存，批大小由 EMBEDDING_BATCH_SIZE 控制
    vectors = await provider.embed_documents([c.text for c in chunks])
    written = await vector_store.upsert_chunks(chunks, vectors)

    numbered = sum(1 for c in chunks if c.clause_no)
    logger.info(f"入库完成 {doc.filename} | 块={len(chunks)}（带条款号 {numbered}）写入={written}")
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "chunks": len(chunks),
        "numbered": numbered,
        "written": written,
    }


async def index_path(path: str, *, strategy: str = "clause") -> dict:
    """从文件路径一路入库。"""
    doc = await asyncio.to_thread(parse_document, path)
    return await index_document(doc, strategy=strategy)
