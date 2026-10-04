"""入库编排：解析 -> 切分 -> 向量化 -> 写 Milvus。

为什么单独成文件
    vector_store 只管 Milvus，chunker 只管切分，embedding 只管编码。
    「把它们串起来」是第三种职责。
    W3 的幂等键、W4 的模型对比都会长在这里，所以值得有自己的位置。
"""

from app.core.chunker import chunk_document
from app.core.embedding import get_embedding_provider
from app.core.logger import logger
from app.core.parser import parse_document
from app.core.schemas import ParsedDocument
from app.core.vector_store import vector_store


async def index_document(doc: ParsedDocument, *, strategy: str = "clause") -> dict:
    """把一份已解析的文档写入向量库。

    注意 embedding 走的是 get_embedding_provider()，不是直接调 model_runtime ——
    这样切到 API 实现时这里一行都不用改（编码约定 6 的直接收益）。
    """
    chunks = chunk_document(doc, strategy=strategy)
    if not chunks:
        return {"doc_id": doc.doc_id, "chunks": 0, "written": 0}

    provider = get_embedding_provider()
    # 批量编码：CPU 上一次编码太多会吃满内存，批大小由 EMBEDDING_BATCH_SIZE 控制
    vectors = await provider.embed_documents([c.text for c in chunks])
    written = await vector_store.upsert_chunks(chunks, vectors)

    numbered = sum(1 for c in chunks if c.clause_no)
    logger.info(
        f"入库完成 {doc.filename} | 块={len(chunks)}（带条款号 {numbered}）写入={written}"
    )
    return {
        "doc_id": doc.doc_id,
        "filename": doc.filename,
        "chunks": len(chunks),
        "numbered": numbered,
        "written": written,
    }


async def index_path(path: str, *, strategy: str = "clause") -> dict:
    """从文件路径一路入库。"""
    doc = parse_document(path)
    return await index_document(doc, strategy=strategy)
