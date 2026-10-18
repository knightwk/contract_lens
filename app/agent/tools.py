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


async def _search_clauses(query: str, top_k: int = 5) -> str:
    """检索条款，返回给模型看的文本。"""
    hits = await retrieve(query, top_k=top_k)
    if not hits:
        return "没有检索到相关条款。"

    # 为什么返回文本而不是 JSON：
    #   模型对"结构化但可读"的文本理解得更稳，而段落里带着 [标识] 和定位，
    #   它才有东西可引用（这直接对应约定 5）。
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        has_clause = any((h.section, h.clause_no, h.clause_title))
        label = h.location if has_clause else "（无条款定位）"
        page = f"第 {h.page} 页" if h.page is not None else "无页码"
        lines.append(f"[{h.chunk_id}] {label} · {h.doc_id} {page}\n{h.text}")
    return "\n\n".join(lines)


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
        logger.exception(f"工具 {name} 执行失败")
        return f"错误：{name} 执行失败（{type(exc).__name__}: {exc}）。可以换个方式再试。"


def tool_schema_json() -> str:
    """把工具定义序列化，用于调试和写进 prompts。"""
    return json.dumps(TOOLS, ensure_ascii=False, indent=2)
