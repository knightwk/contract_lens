"""A5 验收：checkpointer / 续跑 / 多轮对话。

用法
    python scripts/checkpoint_demo.py

三件要验证的事
    1. 有 checkpointer 时，状态真的落盘了（能 get_state 读出来）
    2. 同一个 thread_id 多轮对话，历史会累积 —— 但**不能出现重复的 system 消息**
    3. 🔴 历史里有没有「assistant 有 tool_calls、却没有对应 tool 消息」的坏记录
       —— 这是 A4 硬上限截断留下的坑，续跑时会让 API 报 400（见 EXP-011）

为什么第 3 条最重要
    "单次运行没事" 和 "续跑会炸" 是同一份数据的两种命运。
    能在跑之前就把坏状态挑出来，比等 400 报错再查快得多。
"""

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.graph import build_graph, run_graph  # noqa: E402

Q1 = "合同什么时候生效"
Q2 = "合同里关于不可抗力的约定是什么？"


def find_broken_tool_pairs(messages: list[dict]) -> list[str]:
    """找出「有 tool_calls 但没有对应 tool 消息」的坏记录。

    规则（API 的硬约束）：
        assistant(tool_calls) 后面必须【紧接】着每个 tool_call 对应的 tool 消息。
    返回问题描述列表；空列表表示历史是干净的。
    """
    problems: list[str] = []
    for i, msg in enumerate(messages):
        calls = msg.get("tool_calls") or []
        if msg.get("role") != "assistant" or not calls:
            continue

        expected = {tc.get("id") for tc in calls if isinstance(tc, dict)}
        got: set[str] = set()
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            got.add(messages[j].get("tool_call_id"))
            j += 1

        missing = expected - got
        if missing:
            problems.append(
                f"第 {i} 条 assistant 有 {len(expected)} 个 tool_call，"
                f"但缺 {len(missing)} 条 tool 消息：{sorted(missing)[:2]}"
            )
    return problems


def summarize(messages: list[dict]) -> None:
    roles = [m.get("role") for m in messages]
    print(f"  消息数 {len(messages)}   角色序列: {roles}")
    print(f"  system 消息数: {roles.count('system')}  ← 多轮时应该只有 1 条")
    print(f"  assistant 消息数: {roles.count('assistant')}")
    print(f"  tool 消息数: {roles.count('tool')}")

    broken = find_broken_tool_pairs(messages)
    if broken:
        print("  🔴 发现不完整历史（续跑会 400）：")
        for p in broken:
            print(f"     - {p}")
    else:
        print("  ✅ 工具调用与 tool 消息配对完整")


async def main() -> int:
    saver = MemorySaver()
    tid = "a5-demo"

    print("=" * 74)
    print("[1/3] 单轮 + checkpointer")
    print("=" * 74)
    state = await run_graph(Q1, checkpointer=saver, thread_id=tid)
    summarize(state["messages"])

    print("\n" + "=" * 74)
    print("[2/3] 从 checkpointer 读回落盘的状态")
    print("=" * 74)
    graph = build_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": tid}}
    snap = await graph.aget_state(config)
    saved = snap.values.get("messages", [])
    print(f"  快照里的消息数: {len(saved)}（应和上一轮一致）")
    print(f"  steps={snap.values.get('steps')}")

    print("\n" + "=" * 74)
    print("[3/3] 同一 thread_id 追加第二轮（多轮对话）")
    print("=" * 74)
    # ⚠️ 关键：多轮时【不能】再传 system + 初始状态。
    #   messages 是追加语义，重复传 system 会得到两条 system 消息。
    #   只传「这一轮新增的 user 消息」，LangGraph 会按 reducer 合并到已有状态。
    second = await graph.ainvoke(
        {"question": Q2, "messages": [{"role": "user", "content": Q2}], "steps": 0, "budget_trace": []},
        config=config,
    )
    summarize(second["messages"])

    print("\n" + "=" * 74)
    print("结论要点（这是 A5 该记录的）")
    print("=" * 74)
    print("  · checkpointer 让状态跨调用存活 —— 手写版要自己实现")
    print("  · 多轮对话 = 同一个 thread_id 累积消息，但【只能传增量】，")
    print("    否则追加语义会导致重复 system 消息")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
