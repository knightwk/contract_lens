"""Agent 的工具层：工具定义 + 工具分发。

为什么单独成文件
    Agent 的循环只关心"我要不要调工具、调哪个、参数是什么"。
    工具**具体怎么执行**是另一件事 —— 分开之后，循环可以用假工具测，工具可以单独测。

工具定义用 OpenAI function calling 的格式
    这是目前的事实标准，LangGraph / LlamaIndex / CrewAI 底层都是这一套。
    学会这个格式，换框架不用重学。

⚠️ 一个容易忽略的设计点
    工具的 `description` 不是给人看的注释，是**给模型看的** ——
    模型完全靠它决定"该不该调这个工具、调哪个"。
    description 写得含糊，模型就会选错工具或该调不调。
"""

import json
from typing import Any

from app.core.logger import logger
from app.core.retriever import retrieve
from app.core.vector_store import vector_store

# ────────────────────────────────────────────────────────────
# 工具定义（给模型看）
#
# 注意每个 description 都在回答三个问题：
#   1. 这个工具能做什么
#   2. 什么时候该用它（而不是别的工具）
#   3. 返回什么
# ────────────────────────────────────────────────────────────
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_clauses",
            "description": (
                "在合同条款库里检索与问题相关的条款原文。"
                "当你需要知道合同里**具体怎么写的**、要找依据、或要核对某条约定时使用。"
                "返回若干条款，每条带：条目ID（用于引用）、章节、条款号、页码、原文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索用的关键词或一句话问题，例如「合同什么时候生效」",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 5。不确定时不要传。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clause_details",
            "description": (
                "在合同条款库里根据条款编号以及文档ID搜索匹配的原文。"
                "当你已经知道条款编号以及文档标识，只是精准获取条款原文时使用。"
                "返回匹配的唯一条款，每条带：条目ID（用于引用）、章节、条款号、页码、原文。"
                "如果没有查询到或者查询到空值，直接返回'该条款无内容。'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_no": {
                        "type": "string",
                        "description": "条款号，只填编号，例如「第十五条」「第六条」。不要带条款标题。",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "可选。用于在多个文档有同名条款时收窄，例如「ff478537646e1277」。",
                    },
                },
                "required": ["clause_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "列出知识库里已入库的文档清单。"
                "当你不确定用户问的合同在不在库里、或需要知道有哪些文档时使用。"
                "不需要任何参数。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ────────────────────────────────────────────────────────────
# 工具执行（真正干活的地方）
# ────────────────────────────────────────────────────────────


def _format_hits(rows: list[dict]) -> str:
    """把检索 / 取块的结果统一格式化成给模型看的文本。

    为什么集中在一处
        模型判断"该引用哪一条"完全依赖这里给出的信息（标识、定位、页码、文档）。
        少一个字段，它就只能反问用户 —— 实测 5/5 次反问，补上 doc_id 后消失（EXP-008）。

        所以这个函数是【工具能力的边界】：它给什么，模型才能用什么。

    参数是 dict 而不是 SearchHit
        因为 get_chunks_by_clause 返回的是 Milvus 原始行（没有 score / source），
        用 dict 两边都能喂。_search_clauses 需要把 SearchHit 转成 dict 再调用。
    """
    if not rows:
        return ""

    lines: list[str] = []
    for row in rows:
        parts = [p for p in (row.get("section"), row.get("clause_no"), row.get("clause_title")) if p]
        label = " · ".join(parts) if parts else "（无条款定位）"
        page = f"第 {row['page']} 页" if row.get("page") is not None else "无页码"
        doc = row.get("doc_id") or "（未知文档）"
        lines.append(f"[{row.get('chunk_id', '')}] {label} · {doc} {page}\n{row.get('text', '')}")
    return "\n\n".join(lines)


async def _search_clauses(query: str, top_k: int = 5) -> str:
    """检索条款，返回给模型看的文本。"""
    hits = await retrieve(query, top_k=top_k)
    if not hits:
        return "没有检索到相关条款。"

    # 为什么返回文本而不是 JSON：
    #   模型对"结构化但可读"的文本理解得更稳，而段落里带着 [标识] 和定位，
    #   它才有东西可引用（这直接对应约定 5）。
    return _format_hits([h.model_dump() for h in hits])


# ══════════════════ A2 关键节点：精确取全文工具 ══════════════════
#
# 你要写三样东西：
#   ① TOOLS 里加一个工具定义（放在 search_clauses 之后）
#   ② 写 async def _get_clause_detail(clause_no: str, doc_id: str | None = None) -> str
#   ③ 在 _HANDLERS 里注册
#
# 一、可用的事实（不用猜）
#
#   rows = await vector_store.get_chunks_by_clause(clause_no, doc_id)   -> list[dict]
#     每行有 chunk_id / doc_id / text / clause_no / clause_title / section / page
#     page 已做哨兵还原（无页码是 None）
#
#   _format_hits(rows) -> str        ← 上面已写好，直接复用
#
#   doc_id 可以从 list_documents 或上一次 search_clauses 的结果里拿到
#
#   ⚠️ _search_clauses 现在还是自己拼字符串（第 88-93 行），
#      建议顺手改成 _format_hits 的版本 —— 否则两处格式会慢慢漂移。
#
# 二、这道题真正的考点（不是写代码）
#
#   ① 参数形状怎么定？
#      只要 clause_no，还是也要 doc_id？注意 clause_no 会重复
#      （附件里也有「第一条」）—— 只给条款号可能返回多条。
#      多条时你要【全部返回】还是【让模型自己收窄】？
#      让模型自己收窄，可以根据文档标识收窄，也可以让模型自己来选一个跟靠近用户意图的。
#
#   ② description 怎么写，模型才知道【什么时候该用你】？
#      它得能区分：
#        search_clauses     —— 我只会描述问题，不知道是哪一条
#        get_clause_detail  —— 我已经知道条款号了，直接给我全文
#      写得含糊，模型就会两个都调，或者该调不调。**这是 A2 的核心观察点。**
#
#   ③ 查不到时返回什么？
#      "没有找到第 X 条" 和 空字符串，模型反应会不一样。想清楚再定。
#      可以直接返回该条款无内容。
#
# 三、验收（不只是跑通）
#
#   python scripts/agent_demo.py "第十五条 合同生效及附件 的原文是什么"
#     -> 期望：模型【直接】调 get_clause_detail，而不是先 search_clauses 搜
#
#   python scripts/agent_demo.py "合同什么时候生效"
#     -> 期望：模型先用 search_clauses（它还不知道是哪一条）
#
#   然后记录四问（A2 的主产品，见 docs/plan_agent_track.md）：
#     1. 同一个问题，它选哪个工具？
#     2. 会不会两个都调、或连调两次同一个？
#     3. 参数给错时怎么反应？
#     4. 查不到时是重试、换工具，还是编造？
#
# ══════════════════════════════════════════════════════════════════
async def _get_clause_detail(clause_no: str, doc_id: str | None = None) -> str:
    rows = await vector_store.get_chunks_by_clause(clause_no=clause_no, doc_id=doc_id)
    if not rows:
        scope = f"（限定文档 {doc_id}）" if doc_id else ""
        return f"没有找到条款「{clause_no}」{scope}的原文。"
    return _format_hits(rows=rows)


async def _list_documents() -> str:
    """列出已入库文档。

    只用 vector_store 的公开方法 —— 之前直接伸手去拿 _get_client() 是设计问题：
    需要 # noqa 去压 lint 警告，本身就是"公开 API 缺了一块"的信号。
    """
    # raise RuntimeError("故意炸一下")
    doc_ids = await vector_store.list_doc_ids()
    if not doc_ids:
        return "知识库里还没有任何文档。"
    return "已入库的文档 ID：\n" + "\n".join(f"- {d}" for d in doc_ids)


_HANDLERS = {
    "search_clauses": _search_clauses,
    "get_clause_details": _get_clause_detail,
    "list_documents": _list_documents,
}


async def call_tool(name: str, args: dict[str, Any]) -> str:
    """执行一个工具调用，永远返回字符串。

    ⚠️ 关键设计：**工具出错不抛异常，而是把错误当成一条观察结果返回。**

    为什么：
        Agent 循环里，工具失败是很常见的（参数错、查不到、上游超时）。
        如果直接抛，整个 Agent 就崩了 —— 而正确的行为是让**模型看到错误**，
        然后它自己决定：换个参数重试？换个工具？还是告诉用户做不到。

        这就是 Agent 和"函数调用"的本质区别：错误也是输入的一部分。
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        logger.warning(f"模型请求了不存在的工具：{name}")
        return f"错误：没有名为 {name} 的工具。可用工具：{list(_HANDLERS)}"

    try:
        return await handler(**args)
    except TypeError as exc:
        # 参数不对 —— 告诉模型正确的用法，让它自己改
        logger.warning(f"工具 {name} 参数错误：{exc}")
        return f"错误：调用 {name} 的参数不对（{exc}）。请检查参数名和类型。"
    except Exception as exc:  # noqa: BLE001
        # 工具是拿【模型生成的参数】调用的，所以这里的异常【默认是输入问题】。
        # 打 exception（带堆栈）会污染监控 —— 看到 traceback 会以为系统故障。
        # 代价：工具自身的真 bug 也会降级成 warning，所以要带全上下文才分得清。
        logger.warning(f"工具 {name} 失败 | args={args} | {type(exc).__name__}: {exc}")
        return f"错误：{name} 执行失败（{type(exc).__name__}）。可以换个方式再试。"


def tool_schema_json() -> str:
    """把工具定义序列化，用于调试和写进 prompts。"""
    return json.dumps(TOOLS, ensure_ascii=False, indent=2)
