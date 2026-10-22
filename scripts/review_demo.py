"""B3 验收：复核 Agent + HITL 的完整链路。

用法
    python scripts/review_demo.py

它跑两个用例，验证两件不同的事：

    用例 A（初稿有越界内容）→ 期望 risk_level=high → 图在 human_gate 前【暂停】
    用例 B（初稿内容属实）  → 期望 risk_level=low  → 图【不经过】human_gate，直接结束

为什么必须两个用例
    只测 A，你只能证明"能暂停"；
    必须测 B，才能证明 **interrupt_before 不会无差别拦截** ——
    这是 HITL 设计对不对的关键：`interrupt_before` 是静态配置，
    它只在那个节点【被走到时】才生效。

⚠️ 这个脚本不假设暂停的表示方式
    它在跑完后打印 state.next，让你【看到】框架实际的暂停形态，
    而不是按我的印象写死判断。
"""

import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

# 用例 A：故意捏造一条不存在的条款 —— 复核员应该抓到"越界"
DRAFT_FABRICATED = """## 合同生效结论

知识库里的三份合同**都在签署之日生效**，且都约定了相同的附件效力条款。

- 养老服务合同第十五条：本合同自甲方签字盖章和乙方、丙方签字之日起生效。
- 研学旅游协议第二十六条：自双方当事人签名或者盖章之日起生效。
- **另有一份服务合同第九十九条明确约定：合同自双方完成备案之日起生效。**

共同点：三份合同都要求备案后才生效。
"""

# 用例 B：只陈述材料里确有的内容
#
# ⚠️ 第一版这里写错了：漏掉了 ff478537646e1277 的第十五条，
#    复核员正确地判了 high（"重大遗漏"）——
#    那不是系统的问题，是**测试用例**的问题。
#    修法：把两份真实的生效条款都引用上，这才是"无遗漏"的初稿。
DRAFT_GROUNDED = """## 合同生效结论

查到的两份合同都有明确的生效条款：

- 养老服务合同（ff478537646e1277）第十五条 合同生效及附件：
  本合同一式三份，甲方、乙方、丙方各执一份，具有同等法律效力。
  本合同自甲方签字盖章和乙方、丙方签字之日起生效。[0d533d5f3de20685]

- 研学旅游协议（6a3aa3255b774506）第二十六条 合同生效：
  本合同一式　份，双方各持　份，具有同等法律效力，自双方当事人签名或者盖章之日起生效。[a3f7d2b5a00c4596]

第三份文档（9a5dd9655f7a5227）在检索结果中没有出现生效条款，我不做推断。
"""


def render_pause(snap) -> None:
    print(f"  暂停节点 state.next = {snap.next}")
    values = snap.values or {}
    # review_status 和 risk_level 必须分开看：
    #   incomplete -> 没查完（系统/流程问题）
    #   high       -> 查完了，发现高危问题（内容问题）
    print(f"  review_status = {values.get('review_status')!r}   risk_level = {values.get('risk_level')!r}")
    findings = values.get("findings") or []
    print(f"  findings（{len(findings)} 条）：")
    for f in findings:
        print(f"    - {str(f)[:120]}")


async def run_case(label: str, draft: str) -> None:
    from app.agent.review_graph import build_review_graph

    print("\n" + "=" * 74)
    print(label)
    print("=" * 74)

    saver = MemorySaver()
    graph = build_review_graph(saver)
    config = {"configurable": {"thread_id": f"b3-{label}"}}

    initial = {
        "question": "合同什么时候生效",
        "draft_report": draft,
        "findings": [],
        "risk_level": "",
        "review_status": "incomplete",
        "confirmed": False,
        "messages": [],
    }

    try:
        await graph.ainvoke(initial, config=config)
    except NotImplementedError as exc:
        print(f"\n👉 还没写：{exc}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ 运行失败：{type(exc).__name__}: {exc}")
        return

    snap = await graph.aget_state(config)
    paused = bool(snap.next)

    if paused:
        print("\n⏸  已暂停，等待人工确认：")
        render_pause(snap)
        print("\n  模拟人工确认后恢复执行……")
        try:
            await graph.ainvoke(None, config=config)
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ 恢复失败：{type(exc).__name__}: {exc}")
            return
        final = await graph.aget_state(config)
        print(f"  恢复后 next = {final.next}   confirmed = {final.values.get('confirmed')}")
    else:
        print("\n▶  未暂停，直接跑完：")
        print(f"  review_status = {snap.values.get('review_status')!r}")
        print(f"  risk_level    = {snap.values.get('risk_level')!r}")
        print(f"  confirmed     = {snap.values.get('confirmed')}")


async def main() -> int:
    print("=" * 74)
    print("B3：复核 Agent + HITL")
    print("=" * 74)

    await run_case("用例 A（初稿含捏造条款，期望 high → 暂停）", DRAFT_FABRICATED)
    await run_case("用例 B（初稿只讲有据可依的内容，期望 low → 不暂停）", DRAFT_GROUNDED)

    print("\n" + "=" * 74)
    print("要看的结论")
    print("=" * 74)
    print("  1. 用例 A 的 state.next 应该指向 human_gate（暂停）")
    print("  2. 用例 B 的 state.next 应该为空（没经过 human_gate）")
    print("  3. 如果 B 也暂停了 → interrupt_before 成了无差别拦截，HITL 设计不成立")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
