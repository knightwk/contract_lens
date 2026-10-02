"""Rerank 能力：本地 / API 双实现，统一接口。"""

from functools import lru_cache
from typing import Protocol

import httpx

from app.config import settings
from app.core.model_runtime import model_runtime


class RerankProvider(Protocol):
    """Rerank 能力契约

    rerank 返回的分数【必须与 documents 按下标一一对应】。
    """

    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...

    async def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


class LocalRerank:
    """本地 Cross-Encoder 精排"""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return await model_runtime.rerank_async(query, documents)

    async def warmup(self) -> None:
        await model_runtime.warmup_reranker()

    async def aclose(self) -> None:
        """本地模型无需关闭。"""
        return


class APIRerank:
    """API 精排

    调用 SiliconFlow 等提供的 /v1/rerank 接口，返回分数。
    各家的请求/响应格式不同。下面的实现是参考了 SiliconFlow 的接口，其他厂商请自行修改。
    """

    def __init__(self) -> None:
        if not settings.rerank_api_base or not settings.rerank_api_key:
            raise ValueError("RERANK_PROVIDER=api 时必须配置 RERANK_API_BASE / RERANK_API_KEY / RERANK_API_MODEL")
        self._url = settings.rerank_api_base.rstrip("/") + "/rerank"
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.model_api_timeout)
        return self._client

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        client = self._get_client()
        resp = await client.post(
            self._url,
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={
                "model": settings.rerank_api_model,
                "query": query,
                "documents": documents,
                "return_documents": False,  # 只要分数，不要返回文档
            },
        )
        resp.raise_for_status()

        # 关键：结果数组不一定按原始顺序返回，必须按 index 排序
        scores = [0.0] * len(documents)
        for item in resp.json().get("result", []):
            idx = item["index"]
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = float(item["relevance_score"])
        return scores

    async def warmup(self) -> None:
        """API 模型无需预热。"""
        return

    async def aclose(self) -> None:
        """API 模型关闭。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache
def get_rerank_provider() -> RerankProvider:
    if settings.rerank_provider == "local":
        return LocalRerank()
    return APIRerank()
