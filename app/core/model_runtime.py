"""本地模型运行时：加载、缓存、预热、线程调度等。

业务代码只通过这里访问模型，不直接访问模型类，避免模型类的复杂依赖和初始化逻辑污染业务代码。
"""

import asyncio
import threading
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings
from app.core.logger import logger


class ModelRuntime:
    """本地模型的加载与调度中心

    生命周期：
    1. import 时设置 torch 线程数（必须早于任何推理）
    2. lifespan 调用 warmup_embedder / warmup_reranker 预热模型
    3. encode_async / rerank_async 提供异步推理接口，内部使用线程池调用同步推理接口
    """

    def __init__(self) -> None:
        # torch 的线程池是首次使用时创建的，设晚了不生效
        torch.set_num_threads(settings.torch_num_threads)

        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

        # 必须使用 threading.Lock。加载发生在【工作线程】里，asyncio.Lock 只能在【事件循环线程】里使用。
        self._load_lock = threading.Lock()

        # 并发闸门（惰性、按事件循环创建）
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None

        self._embedder_warmed = False
        self._reranker_warmed = False

    # ---------- 并发闸门 ----------
    def _semaphore(self) -> asyncio.Semaphore:
        """按事件循环惰性创建。

        asyncio.Semaphore 会绑定到创建它的事件循环上，不能跨事件循环使用。
        这里按事件循环惰性创建，避免在不同事件循环里使用同一个 Semaphore ，否则导致异常。
        """
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem = asyncio.Semaphore(settings.max_concurrent_inference)
            self._sem_loop = loop
        return self._sem

    # ---------- 同步内部实现（只在线程池里跑）----------
    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            with self._load_lock:
                if self._embedder is None:
                    path = settings.embedding_model_path
                    logger.info(f"加载 embedding 模型: {path}")
                    self._embedder = SentenceTransformer(str(path))
        return self._embedder

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    path = settings.rerank_model_path
                    logger.info(f"加载 rerank 模型: {path}")
                    self._reranker = CrossEncoder(str(path), max_length=settings.rerank_max_length)
        return self._reranker

    def _encode_sync(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """同步编码。query 侧加指令前缀，doc 侧不加——BGE 是这么训练的。"""
        if not texts:
            return []

        embedder = self._get_embedder()
        if is_query and settings.embedding_query_instruction:
            texts = [settings.embedding_query_instruction + t for t in texts]

        vectors = embedder.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 归一化后内积 == 余弦，Milvus 可用 IP 度量
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _rerank_sync(self, query: str, documents: list[str]) -> list[float]:
        """同步精排。Cross-Encoder 把 query 和 doc 拼在一起前向，无法预计算。"""
        if not documents:
            return []

        reranker = self._get_reranker()
        scores = reranker.predict(
            [(query, doc) for doc in documents],
            batch_size=settings.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    # ---------- 预热 ----------
    async def warmup_embedder(self) -> None:
        """加载 + 预热 embedding 模型，避免首次调用时的延迟。

        两件事：
        - to_thread: 推理是 CPU 密集型，放到线程池里跑，避免阻塞事件循环。
        - 假推理: 只 load 不 infer，tokenizer / 算子的 lazy init 仍然留给第一个真实请求。
        """
        started = time.perf_counter()
        await asyncio.to_thread(self._encode_sync, ["预热"], is_query=False)
        self._embedder_warmed = True
        logger.info(f"embedding 模型就绪，冷启动 {time.perf_counter() - started:.2f}s")

    async def warmup_reranker(self) -> None:
        """加载 + 预热 rerank 模型，避免首次调用时的延迟。"""
        started = time.perf_counter()
        await asyncio.to_thread(self._rerank_sync, "预热", ["预热"])
        self._reranker_warmed = True
        logger.info(f"rerank 模型就绪，冷启动 {time.perf_counter() - started:.2f}s")

    # ---------- 异步对外接口 ----------
    async def encode_async(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        """异步编码。"""
        if not texts:
            return []

        async with self._semaphore():
            return await asyncio.to_thread(self._encode_sync, texts, is_query=is_query)

    async def rerank_async(self, query: str, docs: list[str]) -> list[float]:
        """异步 rerank。"""
        async with self._semaphore():
            return await asyncio.to_thread(self._rerank_sync, query, docs)

    # ---------- 状态（供 /health 使用）----------
    @property
    def embedder_ready(self) -> bool:
        return self._embedder_warmed

    @property
    def reranker_ready(self) -> bool:
        return self._reranker_warmed

    @property
    def is_ready(self) -> bool:
        """两个本地模型都已加载并预热。"""
        return self._embedder_warmed and self._reranker_warmed

    def status(self) -> dict[str, Any]:
        return {
            "embedder_loaded": self._embedder is not None,
            "embedder_warmed": self._embedder_warmed,
            "reranker_loaded": self._reranker is not None,
            "reranker_warmed": self._reranker_warmed,
            "torch_threads": torch.get_num_threads(),
        }


model_runtime = ModelRuntime()
