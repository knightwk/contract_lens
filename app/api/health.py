"""基础设施健康检查端点。

K8s 探针、负载均衡、Docker healthcheck 会永久引用 /health，
它的生命周期和业务端点不同，不该随业务版本演进。
"""

from typing import Any

from fastapi import APIRouter

from app.config import settings
from app.core.model_runtime import model_runtime

router = APIRouter(tags=["Health"])


def _models_ready() -> bool:
    """只检查【当前 provider 真正需要】的模型。

    provider=api 时本地模型不需要就绪，所以不检查——
    否则 /health 会在 API 模式下永远报 false。
    """
    ready = True
    if settings.embedding_provider == "local":
        ready = ready and model_runtime.embedder_ready
    if settings.rerank_provider == "local":
        ready = ready and model_runtime.reranker_ready
    return ready


@router.get("/health", summary="存活与配置检查")
async def health() -> dict[str, Any]:
    """返回进程存活状态和当前生效的关键配置。

    暴露 provider 是有意为之：排查线上问题时，curl /health
    一眼看到实际生效的配置，比翻部署脚本快得多。
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
        "models_ready": _models_ready(),
        "model_runtime": model_runtime.status(),
    }
