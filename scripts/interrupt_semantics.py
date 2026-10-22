"""验证 interrupt_before 的语义：只在【被走到】的节点上生效。

用法
    python scripts/interrupt_semantics.py

为什么单独写这个脚本
    review_demo.py 把两件事混在一起测了：
        · 框架机制：interrupt_before 是否会无差别拦截
        · 模型判断：复核员把风险判成 high 还是 low
    结果两个用例都被判成 high —— **机制这一层就没测到**。

    这个脚本里没有 LLM、没有检索，只有两个节点和一条条件边：
    它跑两次，唯一变量是「路由走 gate 还是走 end」。

    这样如果两次行为不同，就只可能是 interrupt_before 的语义导致的，
    而不是模型输出的随机性。

设计（和 review_graph.py 同构，但极简）
        a --(router)--> gate --(END)
                     +--> END          <- 低风险直接 END，不经过 gate
    compile(interrupt_before=["gate"])

    期望：
        路由到 gate 的那次 -> 暂停，state.next == ('gate',)
        路由到 end 的那次  -> 直接跑完，state.next == ()
"""

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import operator  # noqa: E402
from typing import Annotated, TypedDict  # noqa: E402

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402


class S(TypedDict):
    # route 由外部传入，用来控制这一次跑哪条路 —— 这就是唯一的变量
    route: str
    trace: Annotated[list[str], operator.add]


async def node_a(state: S) -> dict:
    return {"trace": ["a 执行"]}


async def node_gate(state: S) -> dict:
    return {"trace": ["gate 执行（人工确认后才会到这里）"]}


def router(state: S) -> str:
    return "gate" if state["route"] == "gate" else "end"


def build() -> StateGraph:
    b = StateGraph(S)
    b.add_node("a", node_a)
    b.add_node("gate", node_gate)
    b.set_entry_point("a")
    b.add_conditional_edges("a", router, {"gate": "gate", "end": END})
    b.add_edge("gate", END)
    return b.compile(checkpointer=MemorySaver(), interrupt_before=["gate"])


async def run(route: str) -> None:
    graph = build()
    config = {"configurable": {"thread_id": f"t-{route}"}}
    print(f"\n--- route={route!r} ---")

    await graph.ainvoke({"route": route, "trace": []}, config=config)
    snap = await graph.aget_state(config)
    print(f"  state.next = {snap.next}")
    print(f"  trace      = {snap.values.get('trace')}")

    if snap.next:
        print("  ⏸ 暂停了 -> 恢复执行")
        await graph.ainvoke(None, config=config)
        final = await graph.aget_state(config)
        print(f"  恢复后 next  = {final.next}")
        print(f"  恢复后 trace = {final.values.get('trace')}")
    else:
        print("  ▶ 未暂停（这个节点没被走到）")


async def main() -> int:
    print("=" * 74)
    print("interrupt_before 语义验证（无 LLM，唯一变量是路由）")
    print("=" * 74)

    await run("gate")
    await run("end")

    print("\n" + "=" * 74)
    print("判定")
    print("=" * 74)
    print("  正确：route=gate 时 next 指向 ('gate',) 并暂停；")
    print("        route=end  时 next 为空、trace 里没有 'gate 执行'。")
    print("  若 route=end 也暂停 -> interrupt_before 是无差别拦截，")
    print("     那么 review_graph 里的 HITL 设计就不成立，要换成动态 interrupt()。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
