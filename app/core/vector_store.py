"""Milvus 向量库：collection schema、索引、写入与检索。

设计要点
    1. 稠密向量（bge-base-zh，768 维）负责语义匹配（"甲方" ≈ "委托方"）
    2. **BM25 用 Milvus 2.6 的内置能力**，负责精确匹配（"第十二条"）。
       不需要额外模型产出稀疏向量 —— 省一个模型的加载 / 内存 / 延迟，
       也不用维护 tokenizer 与稀疏向量的对齐逻辑。
    3. 写入走 upsert：chunk_id 是确定性 ID（见 chunker._make_chunk_id），
       重复导入同一份合同不产生副本 —— 这是 W3 幂等的基础。
    4. 所有定位字段都落成标量列：W3 的权限过滤与引用回溯都要读它。

为什么用同步 MilvusClient + asyncio.to_thread
    pymilvus 的同步客户端最成熟。把它丢线程池，对外仍是 async 接口，
    与 core/model_runtime.py 的做法一致：阻塞调用不进事件循环。
    （这个模式用了第二次，它已经是约定了 —— 见 docs/conventions.md）

上面每条设计的完整推演和"反过来会怎样"，见 docs/decisions.md：
D-001 / D-010 / D-016 / D-017 / D-019 / D-020 / D-021 / D-022。
"""

import asyncio
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings
from app.core.logger import logger
from app.core.schemas import Chunk, SearchHit

# 检索时要带回的字段。
# ⚠️ Milvus 只返回你【显式列进 output_fields】的列 —— 没写的字段不会出现在结果里，
#    哪怕它存在库里。这是最常见的一类"数据明明写进去了却取不出来"。
#    （主键例外：它总会返回，但键名随类型变化，见 _to_hits）
_OUTPUT_FIELDS = [
    "doc_id",
    "text",
    "clause_no",
    "clause_title",
    "section",
    "page",
    "char_start",
    "char_end",
]

# 标量列的空值哨兵。Milvus 这里不用 nullable，用哨兵更简单、过滤也更直观。
_NO_CLAUSE = ""
_NO_PAGE = -1


class VectorStoreError(Exception):
    """Milvus 操作失败。"""


class VectorStore:
    def __init__(self) -> None:
        self._client: MilvusClient | None = None
        # 只保护 client 与建表的初始化，不保护每次读写
        self._lock = asyncio.Lock()

    # ---------- 连接 ----------

    @property
    def collection_name(self) -> str:
        return settings.milvus_collection_name

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            logger.info(f"连接 Milvus: {settings.milvus_uri}")
            self._client = MilvusClient(
                uri=settings.milvus_uri,
                timeout=settings.milvus_timeout,
            )
        return self._client

    # ---------- 建表 ----------

    @staticmethod
    def _build_schema() -> Any:
        """声明 collection 的字段结构。

        内置 BM25 的三件套（下面已就位）：
          · VARCHAR + enable_analyzer=True  -> 提供分词后的文本
          · SPARSE_FLOAT_VECTOR             -> 存放 BM25 权重
          · Function(BM25) 接起上面两者      -> 写入时自动生成
        推论：**sparse 不在写入时填值**（原因见 _chunk_to_row）。

        标量字段是「引用可回溯」的落点（docs/conventions.md 约定 5）。
        Milvus 只返回【显式声明过】的列 —— 没声明的列在库里根本不存在，
        事后再加要重建集合 + 全量重灌。所以凡是事后推不出来的 Chunk 字段，
        都要在这里落一列。max_length 按实测长度给，不是凭感觉（见 D-019）。
        """
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

        # 主键：确定性 chunk_id（不是 uuid）—— 见 D-001
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)

        # BM25 的输入：开了 analyzer 的文本字段
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
            analyzer_params={"type": settings.milvus_text_analyzer},
        )

        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)
        # BM25 的输出：由下面的 Function 自动填充
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        # 标量列：引用可回溯 + W3 权限过滤
        schema.add_field("clause_no", DataType.VARCHAR, max_length=64)
        schema.add_field("clause_title", DataType.VARCHAR, max_length=64)
        schema.add_field("section", DataType.VARCHAR, max_length=64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("char_start", DataType.INT64)
        schema.add_field("char_end", DataType.INT64)

        # BM25：把 text 自动转成 sparse 向量。有了它就不需要稀疏模型。
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
                params={},
            )
        )
        return schema

    @staticmethod
    def _build_index_params() -> Any:
        """配置两个索引。向量库没有索引就【无法检索】，不是"慢一点"。

        index_type 与 metric_type 的分工（见 D-017）：
          · index_type  —— 用什么数据结构存
          · metric_type —— 怎么算相似度

        稠密：HNSW + IP。HNSW 查询快、占内存，数据量 < 10 万时合适；
              用 IP 而非 COSINE 的前提是向量已 L2 归一化（见 D-010）。
        稀疏：SPARSE_INVERTED_INDEX + BM25。metric 只能是 BM25 ——
              Function 产出的稀疏向量存的是 BM25 词权重，换别的度量就废了。
        """
        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="IP",
            params={"M": settings.milvus_hnsw_m, "efConstruction": settings.milvus_hnsw_ef_construction},
        )

        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        return index_params

    def _ensure_collection_sync(self, recreate: bool) -> bool:
        # 先构造 schema / index（纯本地计算），再连 Milvus ——
        # 配置写错时在开连接之前就失败，不用先白等一次连接超时。
        schema = self._build_schema()
        index_params = self._build_index_params()

        client = self._get_client()
        exists = client.has_collection(self.collection_name)

        if exists and recreate:
            logger.warning(f"删除并重建集合 {self.collection_name}")
            client.drop_collection(self.collection_name)
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info(f"创建集合 {self.collection_name}（稠密 {settings.embedding_dim} 维 + 内置 BM25）")
            return True

        return False

    async def ensure_collection(self, recreate: bool = False) -> bool:
        """确保集合存在。返回 True 表示本次真的创建了。"""
        async with self._lock:
            return await asyncio.to_thread(self._ensure_collection_sync, recreate)

    # ---------- 诊断 ----------

    def _describe_sync(self) -> dict:
        client = self._get_client()
        if not client.has_collection(self.collection_name):
            return {"exists": False, "collection": self.collection_name}

        info = client.describe_collection(self.collection_name)
        stats = client.get_collection_stats(self.collection_name)
        return {
            "exists": True,
            "collection": self.collection_name,
            "fields": [f["name"] for f in info.get("fields", [])],
            "functions": [f.get("name") for f in info.get("functions", [])],
            "row_count": stats.get("row_count", 0),
        }

    async def describe(self) -> dict:
        return await asyncio.to_thread(self._describe_sync)

    async def stats_row_count(self) -> int:
        """get_collection_stats 报的行数 —— 【近似值】，不要拿它做断言。

        Milvus 的 upsert 是 delete + insert：被覆盖的旧行变成墓碑（tombstone），
        compaction 之前仍被这个统计值算进去（见 D-020）。
        Day 4 实测：同一份文档写 3 次，这里报 30，而实际只有 10 条。
        """
        info = await self.describe()
        return int(info.get("row_count", 0))

    def _count_rows_sync(self) -> int:
        """真实可查询行数（墓碑已被过滤）。断言幂等性必须用这个。"""
        result = self._get_client().query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if result and isinstance(result[0], dict):
            return int(result[0].get("count(*)", 0))
        return 0

    async def count_rows(self) -> int:
        return await asyncio.to_thread(self._count_rows_sync)

    # ---------- 写入 ----------

    @staticmethod
    def _chunk_to_row(chunk: Chunk, vector: list[float]) -> dict:
        """把一个 Chunk 转成 Milvus 的一行。

        · **不含 sparse** —— 它的值由 BM25 Function 在写入时生成，不由我们提供
        · Optional 字段用模块顶部的哨兵：Milvus 标量列不接受 None
          （用 `if ... is not None` 判断，不用 `or` —— 对整数会出错，见 D-018）
        · 键名必须与 _build_schema 里的字段名逐字一致
        """
        return {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "clause_no": chunk.clause_no if chunk.clause_no is not None else _NO_CLAUSE,
            "clause_title": chunk.clause_title if chunk.clause_title is not None else _NO_CLAUSE,
            "section": chunk.section if chunk.section is not None else _NO_CLAUSE,
            "page": chunk.page if chunk.page is not None else _NO_PAGE,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "dense": vector,
        }

    def _upsert_sync(self, rows: list[dict], flush: bool) -> int:
        """分批写入。生产环境不该每条都 flush，这里数据量小，为了方便验证就刷了。"""
        client = self._get_client()
        written = 0
        batch_size = settings.milvus_insert_batch_size

        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            client.upsert(collection_name=self.collection_name, data=batch)
            written += len(batch)

        if flush and written:
            client.flush(self.collection_name)

        return written

    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]], *, flush: bool = True) -> int:
        """把分块写入 Milvus。

        用 upsert 而非 insert：chunk_id 是确定性 ID（见 D-001），
        同一份文档重复导入会命中相同主键并被覆盖，不产生副本 —— W3 幂等的落点。

        ⚠️ 验证幂等性要用 count_rows()，不能用 stats_row_count()（见 D-020）。
        """
        if len(chunks) != len(vectors):
            raise VectorStoreError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0

        rows = [self._chunk_to_row(c, v) for c, v in zip(chunks, vectors, strict=True)]
        return self._upsert_sync(rows=rows, flush=flush)

    # ---------- 检索 ----------

    @staticmethod
    def _to_hits(raw: Any, source: str) -> list[SearchHit]:
        """把 Milvus 的原始返回值转成 SearchHit 列表。

        MilvusClient 返回的形状是 [[{...}, ...]] —— 外层是"每个 query 一组"，
        所以取 raw[0]。

        注意 page 的哨兵还原：库里存 -1 表示"无页信息"，出来要变回 None，
        否则下游会以为这份文档真的有第 -1 页。
        """
        hits: list[SearchHit] = []
        for item in raw[0] if raw else []:
            entity = item.get("entity", {}) or {}
            page = entity.get("page", _NO_PAGE)
            hits.append(
                SearchHit(
                    # VARCHAR 主键在搜索结果里的键名是【字段名】(chunk_id)，不是通用键 id。
                    # 只读 item["id"] 会静默拿到空串，而不会报错（见 D-021）。
                    chunk_id=str(item.get("chunk_id") or item.get("id") or ""),
                    doc_id=entity.get("doc_id", ""),
                    text=entity.get("text", ""),
                    score=float(item.get("distance", 0.0)),
                    clause_no=entity.get("clause_no") or None,
                    clause_title=entity.get("clause_title") or None,
                    section=entity.get("section") or None,
                    page=page if page != _NO_PAGE else None,
                    char_start=int(entity.get("char_start", 0)),
                    char_end=int(entity.get("char_end", 0)),
                    source=source,  # type: ignore[arg-type]
                )
            )
        return hits

    def _search_dense_sync(self, vector: list[float], limit: int, expr: str) -> Any:
        return self._get_client().search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="dense",
            limit=limit,
            filter=expr,
            output_fields=_OUTPUT_FIELDS,
            # ef 是 HNSW 的查询期参数：越大越准、越慢。建索引时用的是 efConstruction。
            search_params={"metric_type": "IP", "params": {"ef": settings.milvus_search_ef}},
        )

    def _search_bm25_sync(self, query_text: str, limit: int, expr: str) -> Any:
        return self._get_client().search(
            collection_name=self.collection_name,
            # ← BM25 检索传的是【原始文本】，不是向量。这一步容易记错。
            data=[query_text],
            anns_field="sparse",
            limit=limit,
            filter=expr,
            output_fields=_OUTPUT_FIELDS,
            search_params={"metric_type": "BM25"},
        )

    async def search_dense(self, vector: list[float], *, limit: int = 20, expr: str | None = None) -> list[SearchHit]:
        raw = await asyncio.to_thread(self._search_dense_sync, vector, limit, expr or "")
        return self._to_hits(raw, "dense")

    async def search_bm25(self, query_text: str, *, limit: int = 20, expr: str | None = None) -> list[SearchHit]:
        raw = await asyncio.to_thread(self._search_bm25_sync, query_text, limit, expr or "")
        return self._to_hits(raw, "bm25")

    def _hybrid_search_sync(self, query_text: str, vector: list[float], limit: int, expr: str) -> Any:
        requests = [
            AnnSearchRequest(
                data=[vector],
                anns_field="dense",
                param={"metric_type": "IP", "params": {"ef": settings.milvus_search_ef}},
                limit=limit,
                expr=expr,
            ),
            AnnSearchRequest(
                data=[query_text],
                anns_field="sparse",
                param={"metric_type": "BM25"},
                limit=limit,
                expr=expr,
            ),
        ]
        return self._get_client().hybrid_search(
            collection_name=self.collection_name,
            reqs=requests,
            ranker=RRFRanker(settings.milvus_rrf_k),
            limit=limit,
            output_fields=_OUTPUT_FIELDS,
        )

    async def hybrid_search(
        self,
        query_text: str,
        vector: list[float],
        *,
        limit: int = 10,
        expr: str | None = None,
    ) -> list[SearchHit]:
        """稠密 + BM25 混合检索，RRF 融合。

        RRF: score(d) = Σ 1 / (k + rank_i(d))，k 取 settings.milvus_rrf_k（默认 60）。

        为什么用 RRF 而不是加权求和：它对两路分数的【量纲不敏感】。
        实测分数范围 dense 0.29~0.54 / bm25 0.59~7.58 / hybrid 0.032~0.033 ——
        直接相加等于让 BM25 单方面说了算（详见 D-022）。

        ⚠️ 实现已就位，但【评估】是 W2 Day 13 的事（§14.1 检索方案对比）。
           在那之前不要下"混合更好"的结论 —— 与 Day 2 的 query 前缀实验是同一个教训。
        """
        raw = await asyncio.to_thread(self._hybrid_search_sync, query_text, vector, limit, expr or "")
        return self._to_hits(raw, "hybrid")

    # ---------- 清理 ----------

    async def drop_collection(self) -> None:
        await asyncio.to_thread(self._get_client().drop_collection, self.collection_name)
        logger.warning(f"已删除集合 {self.collection_name}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


vector_store = VectorStore()
