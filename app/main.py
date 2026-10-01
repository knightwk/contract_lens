"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.router import api_router
from app.config import settings
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放。

    会在这里加载并预热本地模型（10-30 秒）。
    放在这里是为了让第一个真实请求不必承担模型冷启动的等待。
    """
    logger.info("-" * 60)
    logger.info(f"{settings.app_name} 启动中 | debug={settings.debug}")
    logger.info(
        f"providers: embedding={settings.embedding_provider} "
        f"rerank={settings.rerank_provider}"
    )

    # 提前把路径打出来： 模型加载失败时，第一眼看这里
    if settings.embedding_provider == "local":
        logger.info(f"embedding 模型路径: {settings.embedding_model_path}")
    if settings.rerank_provider == "local":
        logger.info(f"rerank 模型路径: {settings.rerank_model_path}")

    # TODO: 预热本地模型
    # from app.core.model_runtime import model_runtime
    # await model_runtime.warmup()

    logger.info(f"{settings.app_name} 启动完成，准备接收请求")
    yield

    # TODO: 释放 Milvus 连接、Redis 连接池
    logger.info(f"{settings.app_name} 已关闭")


app = FastAPI(
    title=settings.app_name,
    description="受监管文档智能审查 Agent 平台",
    version="0.1.0",
    lifespan=lifespan,
    # 仅在 debug 模式暴露 API 文档
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# 基础设施端点：不带版本前缀
app.include_router(health_router)

# 业务端点：带版本前缀
app.include_router(api_router, prefix="/api/v1")
