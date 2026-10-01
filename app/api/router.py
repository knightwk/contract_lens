"""业务 API 路由汇总，由 main.py 挂载到 /api/v1 前缀下。"""

from fastapi import APIRouter

api_router = APIRouter()

# 按周次逐步挂载：
# from app.api.v1 import documents, review, reports
# api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
# api_router.include_router(review.router,    prefix="/review",    tags=["Review"])
# api_router.include_router(reports.router,   prefix="/reports",   tags=["Reports"])
