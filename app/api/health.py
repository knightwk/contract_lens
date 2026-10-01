"""基础设施健康检查端点。刻意不放在 /api/v1 下，见下方说明。"""

from typing import Any

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="存活与配置检查")
async def health() -> dict[str, Any]:
    """返回进程存活状态和当前生效的关键配置。

    暴露 provider 是有意为之：排查线上问题时，
    `curl /health` 一眼看到实际生效的配置，比翻部署脚本快得多。
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "debug": settings.debug,
        "providers": {
            "embedding": settings.embedding_provider,
            "rerank": settings.rerank_provider,
        },
        "models": {
            "embedding": settings.embedding_model_name,
            "rerank": settings.rerank_model_name,
            "dim": settings.embedding_dim,
        },
        # Day 2 接上 model_runtime 后改为真实状态
        "models_ready": False,
    }
