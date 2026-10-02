"""Embedding 能力：本地 / API 双实现，统一接口。

provider 的选择发生在这里。model_runtime 只负责本地模型的加载与调度，不关心 provider。
"""

from functools import lru_cache
from typing import Protocol

import httpx

from app.config import settings
from app.core.model_runtime import model_runtime


class EmbeddingProvider(Protocol):
    """Embedding 能力契约

    embed_query 接收单条文本，embed_documents 接收多条文本。
    """

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


class LocalEmbedding:
    """本地模型实现。"""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对文档进行向量化。"""
        return await model_runtime.encode_async(texts, is_query=False)

    async def embed_query(self, text: str) -> list[float]:
        """对查询进行向量化。"""
        embeddings = await model_runtime.encode_async([text], is_query=True)
        return embeddings[0]

    async def warmup(self) -> None:
        """加载 + 预热本地模型，避免首次调用时的延迟。"""
        await model_runtime.warmup_embedder()

    async def aclose(self) -> None:
        """本地模型无需关闭。"""
        return


class APIEmbedding:
    """API 模型实现。"""

    def __init__(self) -> None:
        if not settings.embedding_api_base or not settings.embedding_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=api 时必须配置 EMBEDDING_API_BASE / EMBEDDING_API_KEY / EMBEDDING_API_MODEL"
            )
        self._url = settings.embedding_api_base.rstrip("/") + "/embeddings"
        # 复用一个 client 以保留连接池，避免每次请求都重新建立 TCP 连接
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.model_api_timeout)
        return self._client

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """调用 API 进行向量化。"""
        client = self._get_client()
        resp = await client.post(
            self._url,
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            json={"model": settings.embedding_api_model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])  # 按原始顺序排序
        return [item["embedding"] for item in data]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """对文档进行向量化。"""
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        """对查询进行向量化。"""
        return (await self._embed([text]))[0]

    async def warmup(self) -> None:
        """API 模型无需预热。"""
        return

    async def aclose(self) -> None:
        """关闭 HTTP 连接池。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """根据配置返回 EmbeddingProvider 实例。lru_cache 保证全局只有一个实例，避免重复加载模型或重复创建 HTTP 连接池。"""
    if settings.embedding_provider == "local":
        return LocalEmbedding()
    return APIEmbedding()
