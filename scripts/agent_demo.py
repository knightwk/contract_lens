"""A1 验收：跑手写的 ReAct 循环。

用法
    python scripts/agent_demo.py "合同什么时候生效"
    python scripts/agent_demo.py "这份合同有哪些违约责任条款，分别是第几条？"
    python scripts/agent_demo.py --tools          # 只打印工具定义，不调模型

为什么要打印中间的每一步
    Agent 和普通 RAG 的区别就在"它自己决定调什么"。
    如果不把每一步打出来，你只能看到最后答案 —— 那样根本分不清
    是 Agent 变聪明了，还是它蒙对了。**过程可见是调试 Agent 的前提。**
"""

import argparse
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.react import MAX_STEPS, run_agent  # noqa: E402
from app.agent.tools import tool_schema_json  # noqa: E402

DEFAULT_QUESTION = "合同什么时候生效"


def render_trace(run) -> None:
    """把消息历史渲染成可读的执行轨迹。"""
    print("\n" + "-" * 74)
    print("执行轨迹")
    print("-" * 74)

    step = 0
    for msg in run.messages:
        role = msg.get("role")
        if role == "system":
            continue

        if role == "user":
            print(f"\n[问题] {str(msg.get('content'))[:100]}")

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
            print(f"           [工具返回] {content[:110]}…")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--tools", action="store_true", help="只打印工具定义")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    if args.tools:
        print(tool_schema_json())
        return 0

    print("=" * 74)
    print(f"问题：{args.question}")
    print(f"max_steps：{args.max_steps}")
    print("=" * 74)

    run = await run_agent(args.question, max_steps=args.max_steps)

    render_trace(run)

    print("\n" + "-" * 74)
    print("最终回答")
    print("-" * 74)
    print(run.answer)

    s = run.summary()
    print("\n" + "-" * 74)
    print(f"步数 {s['steps']} · 步数变化 {s['budget_trace']} · 工具调用 {s['tool_calls']} 次 · 用过 {s['tools_used']}")
    print("-" * 74)

    # 判定要区分三件事，缺一不可：
    #   ① 用了工具吗（不用工具 = 那是 RAG，不是 Agent）
    #   ② 给出结论了吗
    #   ③ 是在步数预算内给的吗（兜底文案也是非空，只看 answer 会假阳性）
    # ⚠️ tool_calls == 0 有【两个完全不同的原因】，单一信号区分不了：
    #   (a) 有预算却不调工具、凭记忆作答  -> 那是 RAG 退化，失败
    #   (b) 预算一开始就是 0（max_steps 太小）-> 模型被禁止调工具，属于【预期退化】
    # 之前把 (b) 也判成 (a) —— 判定信号不足以区分情况（同 D-027 那类误判）。
    no_tool_budget = bool(s["budget_trace"]) and min(s["budget_trace"]) <= 0

    if s["tool_calls"] == 0 and no_tool_budget:
        print("ℹ️ 工具预算为 0（max_steps 太小）—— 模型无法调用工具，这条不算失败。")
        print("   要看的是：它有没有【拒绝编造】，而不是拿行业惯例硬答。")
        ok = True
    elif s["tool_calls"] == 0:
        print("❌ 有工具预算却一次都没调用就作答 —— 那是 RAG，不是 Agent")
        ok = False
    elif s.get("budget_exhausted"):
        print(f"❌ 步数用尽（{s['steps']} 步）仍未得出结论 —— 循环控制需要改进（A3）")
        ok = False
    elif not run.answer:
        print("❌ 没有答案")
        ok = False
    else:
        print("✅ 有工具调用，且在预算内给出了答案")
        ok = True

    if s["tool_calls"] > s["steps"]:
        print(f"ℹ️ 工具调用次数（{s['tool_calls']}）多于步数（{s['steps']}）—— 说明有并行调用")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except NotImplementedError as exc:
        print("\n" + "=" * 74)
        print(f"👉 还没写：{exc}")
        print("=" * 74)
        raise SystemExit(2) from None
