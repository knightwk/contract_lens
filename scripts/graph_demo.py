"""A4 验收：LangGraph 版本 vs 手写版本。

用法
    python scripts/graph_demo.py "合同什么时候生效"
    python scripts/graph_demo.py "合同什么时候生效" --both     # 两个版本都跑，对比

为什么必须两个版本都跑
    A4 的目的不是"用框架实现一遍"，而是回答：
        **LangGraph 帮你做了什么？多做了什么？少了什么？**
    这个答案只能靠【同一批问题、两个实现】对比出来 —— 光看代码看不出来。
"""

import argparse
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.react import run_agent  # noqa: E402

DEFAULT_QUESTION = "合同什么时候生效"


def extract_answer(messages: list[dict]) -> str:
    """从消息历史里取最终答案：最后一条【没有 tool_calls】的 assistant 消息。

    为什么从 messages 取，而不是读 state["answer"]
        答案和消息历史是【同一份信息】。在 state 里再存一份 answer 就是影子状态，
        只要存两份就必然有一次忘记同步 —— 实测就出现过"轨迹里有思考、最终回答是空的"。

        **状态里不要存可以推导出来的东西。**

    为什么要求「没有 tool_calls」
        只调工具那一轮的 assistant 消息 content 是 None（或空的），
        那不是答案。真正的答案出现在模型决定收尾的那一轮。
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            text = (msg.get("content") or "").strip()
            if text:
                return text
    return ""


def render_trace(messages: list[dict], question: str) -> None:
    print("\n" + "-" * 74)
    print("执行轨迹（来自 graph 的最终 state）")
    print("-" * 74)

    step = 0
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue

        if role == "user":
            text = str(msg.get("content") or "")
            # 第一轮的 user 消息就是原始问题，不重复打印
            if text != question:
                print(f"\n[注入提示] {text[:110]}")

        elif role == "assistant":
            step += 1
            text = (msg.get("content") or "").strip()
            calls = msg.get("tool_calls") or []
            if text:
                print(f"\n[第 {step} 步 · 思考] {text[:160]}")
            for tc in calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                print(f"[第 {step} 步 · 调用] {fn.get('name')}({fn.get('arguments')})")

        elif role == "tool":
            content = str(msg.get("content") or "").replace("\n", " ")
            print(f"           [工具返回] {content[:100]}…")

    return


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--both", action="store_true", help="同时跑手写版本做对比")
    args = ap.parse_args()

    print("=" * 74)
    print(f"问题：{args.question}")
    print("=" * 74)

    # ---------- LangGraph 版本 ----------
    from app.agent.graph import MAX_STEPS, run_graph

    try:
        state = await run_graph(args.question)
    except NotImplementedError as exc:
        print("\n👉 还没写：", exc)
        return 2

    messages = state.get("messages", [])
    render_trace(messages, args.question)
    answer = extract_answer(messages)

    print("\n" + "-" * 74)
    print("最终回答（graph）")
    print("-" * 74)
    print(answer or "（空 —— 见下方判定）")

    g_steps = state.get("steps", 0)
    g_budget = state.get("budget_trace", [])
    g_ids = state.get("chunk_ids", [])
    print("\n" + "-" * 74)
    print(f"steps={g_steps}  budget_trace={g_budget}  累计见过 {len(g_ids)} 条")
    print("-" * 74)

    # 判定：不能只看"有没有非空字符串"，要区分「答出来了」和「预算耗尽」
    if answer:
        print("✅ 图版本给出了答案")
        ok = True
    elif g_steps >= MAX_STEPS:
        print(f"❌ 步数用尽（{g_steps}/{MAX_STEPS}）仍未得出结论 —— 检查条件边里的硬上限")
        ok = False
    else:
        print("❌ 没有答案，且不是预算耗尽 —— 检查 answer 提取或条件边")
        ok = False

    # ---------- 手写版本对比 ----------
    if args.both:
        print("\n" + "=" * 74)
        print("手写版本（对比）")
        print("=" * 74)
        run = await run_agent(args.question)
        s = run.summary()
        print(f"steps={s['steps']}  budget_trace={s['budget_trace']}")
        print(f"tools={s['tools_used']}")
        print("\n" + "-" * 74)
        print("最终回答（手写）")
        print("-" * 74)
        print(run.answer)

        print("\n" + "=" * 74)
        print("对比（这两行是 A4 的产出，自己填进 EXP-011）")
        print("=" * 74)
        print(f"  步数        : 图 {g_steps}   vs  手写 {s['steps']}")
        print(f"  budget_trace: 图 {g_budget}   vs  手写 {s['budget_trace']}")
        if g_budget and s["budget_trace"]:
            print(f"  注意首轮值是否一致 —— 差一说明「remaining 用进入时的 steps 还是 +1 后的」不同")
        print("  多出来哪些概念：Reducer / State / 条件边 / 路由表 / 编译")
        print("  少了什么      ：步数上限要自己加（手写版 for 免费给的）")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
