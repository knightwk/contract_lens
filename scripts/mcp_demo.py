"""B1 验收：检查 MCP server 暴露了什么（不需要真实客户端）。

用法
    python scripts/mcp_demo.py

为什么不用真实客户端验证
    MCP 的价值在协议层，但验证协议的**形状**不需要跑协议：
    直接问 server「你有哪些 tool / resource、它们的 schema 长什么样」就够了。
    真正的端到端（Claude Desktop 挂载）放到最后手工试一次即可。

这个脚本最重要的作用
    把 **MCP 自动生成的 schema** 和 **tools.py 里手写的 schema** 摆在一起对比。
    你会看到：同一件事，手写要多写 20 行，而且能写错；注解版不会写错。
"""

import asyncio
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _dump(title: str, obj) -> None:
    print("\n" + "-" * 74)
    print(title)
    print("-" * 74)
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


async def main() -> int:
    from app.mcp.server import server

    print("=" * 74)
    print("MCP server 暴露的能力")
    print("=" * 74)

    # ---------- tools ----------
    tools = await server.list_tools()
    print(f"\n[tools] 共 {len(tools)} 个")
    for t in tools:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else "?")
        desc = getattr(t, "description", None) or (t.get("description", "") if isinstance(t, dict) else "")
        print(f"  · {name}")
        print(f"      {str(desc)[:90]}")

    if tools:
        # first = tools[0]
        for i, tool in enumerate(tools):
            # pydantic 模型或 dict，两种都兜住
            payload = tool.model_dump() if hasattr(tool, "model_dump") else tool
            _dump(f"第 {i} 个 tool 的完整定义（MCP 自动生成的 schema）", payload)

    # ---------- resources ----------
    try:
        resources = await server.list_resources()
        print(f"\n[resources] 共 {len(resources)} 个")
        for r in resources:
            name = getattr(r, "name", None) or (r.get("name") if isinstance(r, dict) else "?")
            uri = getattr(r, "uri", None) or (r.get("uri") if isinstance(r, dict) else "")
            print(f"  · {name}  <{uri}>")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[resources] 读取失败：{type(exc).__name__}: {exc}")

    # ---------- 和手写 schema 对照 ----------
    print("\n" + "=" * 74)
    print("对照：手写 schema（tools.py）vs 自动生成（MCP）")
    print("=" * 74)

    from app.agent.tools import TOOLS

    hand = TOOLS[0]["function"]
    print(f"\n手写版：{hand['name']}")
    print(f"  parameters 行数：{json.dumps(hand['parameters'], ensure_ascii=False, indent=2).count(chr(10)) + 1}")
    hand_params = json.dumps(hand["parameters"], ensure_ascii=False)
    print(f"  有没有 'parameters' 键：{'parameters' in hand}")

    if tools:
        auto = tools[0]
        auto_payload = auto.model_dump() if hasattr(auto, "model_dump") else auto
        auto_str = json.dumps(auto_payload, ensure_ascii=False, default=str)
        print(f"\n自动版：{getattr(auto, 'name', '?')}")
        print(f"  行数：{auto_str.count(chr(10)) + 1}")
        print("  —— 关键在于：手写版要人肉保证 type 是 'string'、键名是 'parameters'，")
        print("     注解版由 Python 类型系统保证。**这类坑不再可能。**")

    print("\n" + "=" * 74)
    print("下一步：端到端（可选，手工）")
    print("=" * 74)
    print("  python -m app.mcp.server stdio        # 供 Claude Desktop 等本地客户端挂载")
    print("  python -m app.mcp.server http         # streamable-http，监听 127.0.0.1:8765")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except NotImplementedError as exc:
        print("\n👉 还没写：", exc)
        raise SystemExit(2) from None
