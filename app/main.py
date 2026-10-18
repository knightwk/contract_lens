"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.router import api_router
from app.config import settings
from app.core.embedding import get_embedding_provider
from app.core.llm import llm_client
from app.core.logger import logger
from app.core.reranker import get_rerank_provider
from app.core.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放。

    三笔冷启动开销都在这里付掉，不让第一个真实请求替所有人买单：
      1. 本地模型加载（embedding / rerank）
      2. Milvus 建连（实测首次 4.1s）
      3. tiktoken 词表加载（实测 1~2s，藏在 count_messages_tokens 里，不打点根本看不见）
    """
    logger.info("-" * 60)
    logger.info(f"{settings.app_name} 启动中 | debug={settings.debug}")
    logger.info(f"providers: embedding={settings.embedding_provider} rerank={settings.rerank_provider}")

    embedding_provider = get_embedding_provider()
    rerank_provider = get_rerank_provider()

    try:
        await embedding_provider.warmup()
        await rerank_provider.warmup()
        # 顺带确认集合存在 —— 连不上 Milvus 时在这里就失败，而不是等第一个上传请求
        await vector_store.ensure_collection()
    except Exception:
        logger.exception("核心依赖预热失败，服务不启动")
        # 模型与向量库都是这个服务的核心能力，预热失败就不要对外提供服务（fail fast）
        raise

    # tiktoken 词表加载失败不致命：count_tokens 内部会退化为按字符数估算
    llm_client.count_tokens("预热")

    logger.info(f"{settings.app_name} 启动完成，准备接收请求")
    yield

    await embedding_provider.aclose()
    await rerank_provider.aclose()
    vector_store.close()
    await llm_client.aclose()
    # TODO: 释放 Redis 连接池
    logger.info(f"{settings.app_name} 已关闭")


app = FastAPI(
    title=settings.app_name,
    description="受监管文档智能审查 Agent 平台",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.debug else None,
)


app.include_router(health_router)  # 基础设施端点：不带版本前缀
app.include_router(api_router, prefix="/api/v1")  # 业务端点：带版本前缀
