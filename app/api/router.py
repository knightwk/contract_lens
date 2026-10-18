"""业务 API 路由汇总，由 main.py 挂载到 /api/v1 前缀下。"""

from fastapi import APIRouter

from app.api.v1 import documents, query

api_router = APIRouter()

api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
# ⚠️ query 是【动作型】端点，不是资源集合 —— 不能再加 /query 前缀，
#    否则路径会变成 /api/v1/query/query（路由里的 path 已经是 /query 了）。
api_router.include_router(query.router, tags=["Query"])

# 按周次逐步挂载：
# from app.api.v1 import review, reports
# api_router.include_router(review.router,  prefix="/review",  tags=["Review"])
# api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
