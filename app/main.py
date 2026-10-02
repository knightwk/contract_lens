"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.router import api_router
from app.config import settings
from app.core.embedding import get_embedding_provider
from app.core.logger import logger
from app.core.reranker import get_rerank_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放。

    会在这里加载并预热本地模型，为了让第一个真实请求不必承担模型冷启动的等待。
    """
    logger.info("-" * 60)
    logger.info(f"{settings.app_name} 启动中 | debug={settings.debug}")
    logger.info(f"providers: embedding={settings.embedding_provider} rerank={settings.rerank_provider}")

    embedding_provider = get_embedding_provider()
    rerank_provider = get_rerank_provider()

    try:
        await embedding_provider.warmup()
        await rerank_provider.warmup()
    except Exception:
        logger.exception("模型预热失败，服务不启动")
        # 模型是这个服务的核心能力，预热失败就不要对外提供服务（fail fast）
        raise

    logger.info(f"{settings.app_name} 启动完成，准备接收请求")
    yield

    await embedding_provider.aclose()
    await rerank_provider.aclose()
    # TODO: 释放 Milvus 连接、Redis 连接池
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
