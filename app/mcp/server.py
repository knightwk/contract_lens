"""ContractLens 的 MCP server —— 把知识库能力暴露给任何 MCP 客户端。

为什么值得做
    MCP 是「工具的标准接口」。做完之后 Claude Desktop / Cursor / 任何 MCP 客户端
    都能直接挂载你的合同知识库 —— 不用为每个客户端重写一遍集成。

    JD 里出现两次（JD2 的 MCP/Skills、JD3 的 FastMCP），是这个阶段 ROI 最高的一件事。

════════════════════════════════════════════════════════════════
⚠️ 版本注意：FastMCP 在 mcp 2.x 里改名了

    v1:  from mcp.server.fastmcp import FastMCP
    v2:  from mcp.server.mcpserver import MCPServer      ← 本项目用这个

    本项目选 v2 是因为「无状态 HTTP」只在 v2 里实现（stateless_http=True）。
    面试时可以主动提这个改名 —— 说明你在跟进版本，而不是只会照抄教程。

🔑 和手写 function calling 的最大差别

    app/agent/tools.py 里是【手写 JSON schema】：
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
    手写就会写错 —— 我们踩过 `parameter`（少个 s）、`"str"`（不是 string）
    这类坑，而且**不会报错，只会让工具静默失效**。

    MCP 这边是【装饰器 + 类型注解】：
        类型注解 -> parameters 自动生成
        docstring -> description 自动生成
    **那类坑在这里根本不会发生。** 这才是"标准协议"的实际价值。
════════════════════════════════════════════════════════════════
"""

import json
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from app.core.logger import logger
from app.core.retriever import retrieve
from app.core.vector_store import vector_store

server = MCPServer("contract-lens")


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


# ══════════════════ B1 的设计决定（已定，见 D-035） ══════════════════
#
# ① tool 还是 resource？—— 判据不是「谁决定调用」
#
#   常见误解：tool 由模型决定、resource 由客户端决定。
#   但现代 MCP 客户端里，模型同样可以决定去读一个 resource（用户 @ 引用、
#   或客户端把资源列表交给模型挑）。**「谁决定」两边都成立，不是判据。**
#
#   真正的判据是「动作 vs 数据」：
#     tool     = 函数（动作）：有参数、执行后有结果、不能被缓存
#     resource = 数据（内容）：靠 URI 定位、**无参数**、可缓存可订阅
#
#   按这把尺子：search_clauses / get_clause_details 有参数、是动作 -> tool
#               list_documents 返回的是一份"目录"数据、无参数 -> 倾向 resource
#
#   但还有一个更实用的判据：**谁在消费它**
#     消费者是 Agent（tool-calling 循环）-> 必须是 tool
#       因为循环里【只有 tools】。resource 进不了 tool_calls，模型无法主动读，
#       只能靠客户端把资源塞进上下文 —— 那 Agent 就没法自己决定"先看看有哪些文档"。
#     消费者是聊天客户端 -> resource 更合适
#       用户可以 @ 引用、能列在侧栏、能缓存、能订阅变更。
#
#   **本项目的实际做法（注意和"理想做法"不同）：**
#
#     对外（任何 MCP 客户端）：documents 注册为 **resource**
#     对内（本项目的 Agent）：用 app/agent/tools.py 里的 list_documents **tool**
#     两者【不走同一条路】—— Agent 不经过 MCP，直接调底层函数。
#
#   为什么不让 Agent 也走 MCP？
#     少一层协议开销（进程间 JSON-RPC），调试也直观。代价是：
#
#     ⚠️ **工具描述现在有两份**：tools.py 的 JSON schema 和 server.py 的
#     `Annotated + Field`。改一处忘另一处，两份就会漂移 ——
#     和当初 _format_hits 那个"两处格式会慢慢漂移"是同一类问题。
#
#     两条收敛路线，选一条写进 decisions：
#       (a) 让 Agent 通过 MCP 客户端消费自己的 server（统一接口，多一层开销）
#       (b) 把工具定义抽到共享模块，tools.py 和 server.py 都从那里生成
#     当前 TODO：先记着这个债，B 周结束前处理。
#
# ② 暴露几个？—— 现在全部暴露，但这是有前提的
#
#   前提：**当前三个能力全部只读、无副作用。**
#   一旦加写操作（导出报告 / 提交审核 / 修改条款），必须重新逐项评估，
#   不能沿用"全暴露"这个结论。
#
#   ⚠️ 一个现在就已经存在的风险：
#     **server 没有任何鉴权。** 任何能连上的客户端都能拿到全部 doc_id，
#     再逐份读全文 —— 等于把整个知识库对外开放。
#     单用户项目无所谓，但多租户下客户 A 看到客户 B 的文档 ID 就是泄露。
#
#     生产上应该在 tool 层按 doc_id 做权限过滤，而不是指望客户端自律。
#     MCP 把「能力」标准化了，同时也把「暴露面」标准化了 —— 这是一个副作用。
# ══════════════════════════════════════════════════════════════
#
# 实现（模式已确定，照抄形状即可）
#
#   工具：
@server.tool()
async def search_clauses(
    query: Annotated[str, Field(description="检索用的关键词或一句话问题，例如「合同什么时候生效」")],
    top_k: Annotated[int, Field(description="返回条数，默认 5。不确定时不要传。")] = 5,
) -> str:
    """在合同条款库里检索与问题相关的条款原文。

    当需要知道合同里【具体怎么写的】、要找依据时使用。
    返回条款，每条带：条目ID、章节、条款号、页码、原文、所属文档。
    """
    hits = await retrieve(query=query, top_k=top_k)
    if not hits:
        return "没有检索到相关条款"

    return _format_hits([h.model_dump() for h in hits])


@server.tool()
async def get_clause_details(
    clause_no: Annotated[str, Field(description="条款号，只填编号，例如「第十五条」「第六条」。不要带条款标题。")],
    doc_id: Annotated[
        str | None, Field(description="可选。用于在多个文档有同名条款时收窄，例如「ff478537646e1277」。")
    ] = None,
) -> str:
    """在合同条款库里根据条款编号以及文档ID搜索匹配的原文。

    当你已经知道条款编号以及文档标识，只是精准获取条款原文时使用。
    返回匹配的唯一条款，每条带：条目ID（用于引用）、章节、条款号、页码、原文。
    如果没有查询到或者查询到空值，直接返回'该条款无内容。'。
    """
    rows = await vector_store.get_chunks_by_clause(clause_no=clause_no, doc_id=doc_id)
    if not rows:
        scope = f"（限定文档 {doc_id}）" if doc_id else ""
        return f"没有找到条款「{clause_no}」{scope}的原文。"
    return _format_hits(rows=rows)


#   资源：
@server.resource("contractlens://documents", name="documents", mime_type="application/json")
async def documents() -> str:
    """列出知识库里已入库的文档清单。

    "当你不确定用户问的合同在不在库里、或需要知道有哪些文档时使用。"
    返回已入库的文档 JSON 格式信息。
    """
    doc_ids = await vector_store.list_doc_ids()
    if not doc_ids:
        return json.dumps({"count": 0, "documents": []}, ensure_ascii=False)
    return json.dumps({"count": len(doc_ids), "documents": doc_ids}, ensure_ascii=False)


#   ⚠️ 两个要点：
#     · **docstring 就是 description** —— 它决定模型会不会选这个工具，
#       所以要和 tools.py 里那段写得一样用心（写清楚"什么时候用它"）
#     · 返回类型建议 `str`（MCP 会按类型生成 output schema；
#       返回 dict 也可以，但 str 更容易让模型直接读）
#
# 三、验收（不需要真实客户端）
#
#   python scripts/mcp_demo.py
#     -> 打印 server 自动生成的 tool / resource 清单与 schema
#     -> 和 tools.py 里手写的 schema 对照，看差异
#
# ══════════════════════════════════════════════════════════════


def serve(transport: str = "stdio", *, port: int = 8765) -> None:
    """启动 server。

    transport 的选择（也是 B1 要判断的）：
        stdio           —— 本地进程内管道，Claude Desktop 这类客户端用
        streamable-http —— 远程挂载，可以 stateless
        sse             —— 老式流式，一般不新用
    """
    logger.info(f"启动 MCP server：transport={transport}")
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="streamable-http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    import sys

    serve(sys.argv[1] if len(sys.argv) > 1 else "stdio")
