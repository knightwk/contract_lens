"""检索问答接口。

POST /api/v1/query
"""

from fastapi import APIRouter, HTTPException

from app.api.v1.schemas import Citation, QueryRequest, QueryResponse, TimingBreakdown
from app.core.llm import LLMError
from app.core.logger import logger
from app.core.rag import answer, extract_markers
from app.core.schemas import SearchHit
from app.core.timing import Timing

router = APIRouter()


def _citation_location(hit: SearchHit) -> str:
    """拼出给用户看的定位：条款位置 + 页码。

    为什么不能直接 f"{hit.location} 第{hit.page}页"：
        page 可能为 None（Markdown 来源没有页码），f-string 会把 None
        原样渲染成字符串 —— 变成「第None页」。
        同一类问题在 Day 4（_chunk_to_row）和 Day 5（_format_context）各出现过一次，
        见 docs/decisions.md D-018。

    为什么不用 hit.location 兜底：
        没有条款信息时它会退化成 "doc_id#char_start"，那是内部序号，
        对用户没有意义，不如明确写「（无条款定位）」。
    """
    has_clause = any((hit.section, hit.clause_no, hit.clause_title))
    label = hit.location if has_clause else "（无条款定位）"
    page = f"第 {hit.page} 页" if hit.page is not None else "无页码"
    return f"{label} · {page}"


@router.post("/query", response_model=QueryResponse, summary="按问题检索并作答（带引用）")
async def query(req: QueryRequest) -> QueryResponse:
    """检索 → 组装 Prompt → 调 LLM → 返回带引用的回答。

    注意返回的是 citations（回答的依据），不是检索到的全部材料 ——
    用户要的是"我这句话从哪来"，不是检索日志。

    ⚠️ 这个接口有一个前置约定：**回答必须基于原文且带引用**（约定 5）。
       检索不到依据时不硬答，而是返回 grounded=false（见下面的失败路径说明）。
    """
    timing = Timing()

    try:
        result = await answer(req.question, top_k=req.top_k, timing=timing)
    except LLMError as exc:
        logger.error(f"问答失败：{exc}")
        # 502 —— 本服务是好的，出错的是上游 LLM（重试耗尽后仍失败）。
        #   不用 503（那是"本服务不可用"）；
        #   不用 504（只有确属超时才更贴切，而失败原因可能多样）。
        # detail 用结构化对象而不是裸字符串：前端能按 code 分支，
        # W6 的故障实验也能靠它断言"是不是预期的那个失败"。
        raise HTTPException(
            status_code=502,
            detail={"code": "LLM_UPSTREAM_FAILED", "message": str(exc)},
        ) from exc

    # ── 失败路径：Day 6 的四个决定 ──
    #
    # ① 检索为空 -> 200，不是 404/422
    #    用户视角：这是【正常结果】（"知识库里没有"），UI 上该显示成提示，不是红色报错。
    #    语义：/query 这个资源是存在的，只是这次没依据可给 —— 404 说的是"资源不存在"。
    #    422 更不能���：FastAPI 已经用它表示参数校验失败，混在一起前端没法区分。
    #
    #    但 200 会让监控看不见"答不上来"，所以给响应加一个机器可读的 grounded=false，
    #    用来统计"无依据率"。这个数 W6 的结果表和故障实验都要用。
    #
    # ② LLM 失败 -> 502（见上面的 except 分支）
    #
    # ③ 参数校验失败（question 为空 / 超长 / top_k 越界）
    #    Pydantic 自动返回 422，不用手动处理 —— 唯一例外是"业务层"的校验
    #    （比如 W3 的权限拒绝），那种要自己抛。
    #
    # ④ 错误响应体
    #    统一成 {"detail": {"code": "...", "message": "..."}}。
    #    机器读 code，人读 message —— W3 的权限拒绝、W6 的故障实验都靠 code 断言。
    if not result.hits:
        return QueryResponse(
            question=req.question,
            answer="知识库中没有找到与该问题相关的条款。",
            citations=[],
            grounded=False,
            timings=TimingBreakdown(**timing.to_dict()),
        )

    # 只返回【回答实际引用了的】材料，不是检索到的全部 ——
    # 用户要看的是"我这句话从哪来"，不是检索日志。
    cited = set(extract_markers(result.answer))
    citations: list[Citation] = [
        Citation(
            doc_id=h.doc_id,
            chunk_id=h.chunk_id,
            clause_location=_citation_location(h),
            original_text=h.text,
        )
        for h in result.hits
        if h.chunk_id in cited
    ]
    grounded = bool(citations) and "无法回答" not in result.answer

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        citations=citations,
        timings=TimingBreakdown(**timing.to_dict()),
        grounded=grounded,
    )
