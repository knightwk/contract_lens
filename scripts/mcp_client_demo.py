"""B1 端到端：用真实 MCP 客户端连上 server，走一遍完整协议往返。

用法
    python scripts/mcp_client_demo.py            # stdio（自动拉起 server 子进程，推荐）
    python scripts/mcp_client_demo.py --http     # 连已在运行的 streamable-http 服务

为什么用 SDK 客户端而不是 Inspector
    Inspector 要额外装 Node 工具链。用官方 SDK 写 40 行客户端：
      · 走的是【完全一样的协议】，不是模拟
      · 脚本可复现、可进 CI
      · 顺便验证了 stdio 传输本身（那正是 Claude Desktop 用的方式）

⚠️ stdio 传输有一个必须知道的约束
    server 的 **stdout 被 JSON-RPC 独占**。任何 print / 日志写到 stdout
    都会污染协议帧，客户端会解析失败。
    所以：日志必须走 stderr 或文件 —— loguru 默认写 stderr，符合要求。
    但如果你在 server 里加了 print()，stdio 模式就会立刻坏掉。

为什么这个脚本能证明"mime_type 是对的"
    它会真的把 documents 资源读回来并 json.loads() 一次。
    声明 application/json 却返回纯文本的话，这一步会直接抛解析错误。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOCUMENTS_URI = "contractlens://documents"


def _text_of(result) -> str:
    """从 CallToolResult 里取出文本内容。"""
    blocks = getattr(result, "content", None) or []
    parts = []
    for b in blocks:
        text = getattr(b, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def exercise(session) -> bool:
    """在已初始化的会话上跑一遍完整检查。返回是否全过。"""
    ok = True

    # ---------- 1. tools ----------
    print("\n[1/4] list_tools")
    tools = (await session.list_tools()).tools
    for t in tools:
        props = (t.input_schema or {}).get("properties") or {}
        no_desc = [k for k, v in props.items() if not v.get("description")]
        flag = "✅" if not no_desc else f"⚠️ 缺描述 {no_desc}"
        print(f"  · {t.name:20} 参数={list(props)}  {flag}")
    if not tools:
        print("  ❌ 没有任何工具")
        ok = False

    # ---------- 2. resources ----------
    print("\n[2/4] list_resources")
    resources = (await session.list_resources()).resources
    for r in resources:
        print(f"  · {r.name}  <{r.uri}>  mime={r.mime_type}")
    has_docs = any(str(r.uri) == DOCUMENTS_URI for r in resources)
    print(f"  包含 {DOCUMENTS_URI}: {'✅' if has_docs else '❌'}")
    ok = ok and has_docs

    # ---------- 3. 读资源：验证 mime_type 与内容一致 ----------
    print("\n[3/4] read_resource（验证声明的 mime_type 与真实内容一致）")
    if has_docs:
        try:
            read = await session.read_resource(DOCUMENTS_URI)
            raw = ""
            for c in read.contents:
                raw += getattr(c, "text", "") or ""
            print(f"  原始返回（前 120 字）：{raw[:120]}")
            try:
                parsed = json.loads(raw)
                print(f"  ✅ 能按 JSON 解析，键={list(parsed) if isinstance(parsed, dict) else type(parsed).__name__}")
                if isinstance(parsed, dict) and "documents" in parsed:
                    print(f"     文档数：{parsed.get('count')} -> {parsed['documents']}")
            except json.JSONDecodeError as exc:
                print(f"  ❌ 声明 application/json 但解析失败：{exc}")
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ 读资源失败：{type(exc).__name__}: {exc}")
            ok = False

    # ---------- 4. 调工具 ----------
    print("\n[4/4] call_tool")
    cases = [
        ("search_clauses", {"query": "合同什么时候生效", "top_k": 3}),
        ("get_clause_details", {"clause_no": "第十五条"}),
    ]
    for name, args in cases:
        if not any(t.name == name for t in tools):
            print(f"  ⚠️ 跳过 {name}（server 没暴露这个工具）")
            continue
        try:
            result = await session.call_tool(name, args)
            text = _text_of(result)
            is_error = getattr(result, "isError", False)
            mark = "❌" if (is_error or not text) else "✅"
            print(f"\n  {mark} {name}({json.dumps(args, ensure_ascii=False)})")
            print(f"     {'isError' if is_error else '返回'}（前 160 字）：")
            print(f"     {text[:160]}")
            if is_error or not text:
                ok = False
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ❌ {name} 抛异常：{type(exc).__name__}: {exc}")
            ok = False

    return ok


async def via_stdio() -> bool:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server", "stdio"],
        cwd=str(ROOT),
        # 显式 utf-8：Windows 默认可能是 gbk，中文会乱码或抛错
        encoding="utf-8",
    )
    print(f"启动子进程：{sys.executable} -m app.mcp.server stdio")
    print(f"工作目录：{ROOT}")

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            srv = getattr(info, "serverInfo", None)
            print(f"已连接：server={getattr(srv, 'name', '?')} v{getattr(srv, 'version', '?')}")
            return await exercise(session)


async def via_http(url: str) -> bool:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    print(f"连接：{url}")
    async with streamable_http_client(url) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            srv = getattr(info, "serverInfo", None)
            print(f"已连接：server={getattr(srv, 'name', '?')} v{getattr(srv, 'version', '?')}")
            return await exercise(session)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true", help="连已在运行的 streamable-http 服务")
    ap.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    args = ap.parse_args()

    print("=" * 74)
    print("MCP 端到端验证")
    print("=" * 74)

    ok = await (via_http(args.url) if args.http else via_stdio())

    print("\n" + "=" * 74)
    print("✅ 端到端通过：协议往返、工具调用、资源读取、mime_type 一致" if ok
          else "❌ 有检查未通过（见上）")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
