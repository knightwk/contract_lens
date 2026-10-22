"""B3：复核 Agent + 人工确认（HITL）。

════════════════════════════════════════════════════════════════
一、架构选择：为什么是「顺序链」而不是 supervisor

    常见三种多 Agent 形态：

      (a) 顺序链（pipeline）  A 的输出是 B 的输入。固定分工。
      (b) supervisor          一个调度者决定"下一步叫谁"。
      (c) 并行 + 汇总        多个 Agent 同时干，最后合并。

    本项目选 (a)，理由是：
      · **任务固定** —— 永远是"复核审查结果"，不需要动态选人
      · **成本可预测** —— 就是 2 段 LLM 调用，不是不确定的调度树
      · **可测** —— 两个 Agent 能各自单独跑、单独断言

    supervisor 的适用场景是「任务类型不确定」（比如客服里既可能是退款也可能是查询）。
    在这里用它，只会多一层调度 LLM 调用，而且**调度本身可能选错**。

    ⚠️ 面试会问「为什么不用 supervisor」，上面这段就是答案。

二、为什么用两个独立的图，而不是一个图里塞两个 agent 节点

    · 两个 Agent 的 prompt、工具集、终止条件都不同，混在一个图里状态会互相污染
    · 独立图可以**各自单独跑和测**（Agent1 出问题时不牵连 Agent2）
    · 代价是多一次状态传递（draft_report 作为字符串传过去）—— 这个代价很小

三、HITL 的位置

    不是"每一步都停下来"，而是 **只在高风险结论上停**：

        review_node 判定 risk_level
              ├─ "low"  -> 直接收尾
              └─ "high" -> 进入 human_gate（这个节点被 interrupt_before 拦住）

    这样 `interrupt_before=["human_gate"]` 只在【该节点被走到时】才生效，
    低风险路径根本不会经过它。

    ⚠️ 这个"只在被走到时触发"的行为是设计意图，**必须实测确认** ——
       见 scripts/review_demo.py 的低风险用例。
════════════════════════════════════════════════════════════════
"""

import json
import operator
import re
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.tools import TOOLS, call_tool
from app.config import settings
from app.core.llm import llm_client

SYSTEM_PROMPT = """你是一个合同审查的复核员。你的任务不是重新回答，而是**找出初稿的问题**。

你要检查三件事：
1. **遗漏** —— 初稿有没有漏掉重要的条款？（用工具去查，不要凭印象）
2. **矛盾** —— 初稿内部有没有前后不一致？两份合同的同一事项是否被说成一样？
3. **越界** —— 初稿有没有说出材料里没有的东西？这是最严重的问题。

工具用法：
- compare_clauses：对比两份合同的同一个条款，看差异
- search_clauses / get_clause_details：核对初稿引用的内容是否属实

最后必须输出一行：
RISK: high 或 RISK: low
判断标准：出现「越界」或「矛盾」= high；只有措辞/完整性问题 = low。
"""


class ReviewState(TypedDict):
    """复核图的状态。

    draft_report 用【覆盖】语义：它只写一次，之后不变。
    findings 用【追加】语义：多轮检查可能累积多条问题。
    """

    question: str
    draft_report: str
    findings: Annotated[list[str], operator.add]
    risk_level: str
    review_status: str
    confirmed: bool
    messages: Annotated[list, operator.add]


# ══════════════════ B3 关键节点 ①：compare_clauses 工具 ══════════════════
#
# 在 app/agent/tools.py 里加一个工具（这是 D-034 里唯一值得加的那个）：
#
#   compare_clauses(doc_a: str, doc_b: str, clause_key: str) -> str
#
# 要回答的设计问题：
#
#   ① 怎么"对比"？
#      最省事的做法是【分别取出两边的条款原文，并排返回】，
#      让模型自己看差异 —— **不要试图在代码里做文本 diff**。
#      为什么：法务条款的差异是语义层面的（"签名或盖章" vs "签名并盖章"），
#      字符 diff 只会给你一堆噪音，模型看原文判断得更准。
#
#   ② 参数该叫什么？
#      clause_key 而不是 clause_no —— 用户可能只记得"生效"这个词，
#      不记得是第几条。所以这里走【关键词匹配 clause_title】，
#      这也让 compare_clauses 和 get_clause_details 的分工更清楚：
#        get_clause_details = 已知条款号，取一份的全文
#        compare_clauses    = 只记得主题词，取两份各自的条款并排
#
#   ③ doc_a / doc_b 不存在时返回什么？
#      明确说"文档 X 不在库里，现有文档是 [...]"，不要返回空串。
#
# ═══════════════════════════════════════════════════════════════════════


# ══════════════════ B3 关键节点 ②：复核节点 ══════════════════
#
# 要做的事：
#   ① 把 state["draft_report"] 作为待复核对象放进 messages
#   ② 跑一个小循环（复用 react.py 的思路，但 1~2 轮足够）
#      —— 复核员的活是"抽查"，不是"重新查一遍"
#   ③ 从模型输出里解析 RISK: high / low -> state["risk_level"]
#   ④ 返回增量（messages / findings / risk_level）
#
# ⚠️ 必须处理的边界：模型可能不按格式输出 "RISK:" 那一行。
#    解析失败时 —— **默认按 high 处理**。
#    理由：复核失败时宁可多拦一次人工，也不要漏掉一个高危结论。
#
# ═══════════════════════════════════════════════════════════════


MAX_ROUNDS = 6

# 最后一轮注入的收口指令。配合"不给 tools"双保险 ——
# 只靠提示词，模型仍可能坚持调工具；不给 tools 才是硬约束。
FINISH_HINT = (
    "你已经用完检索机会。现在必须直接给出复核结论，"
    "并在最后单独一行输出 RISK: high 或 RISK: low。不要再调用工具。"
)


async def review_node(state: ReviewState) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"用户问题：\n{state['question']}\n\n待复核初稿：\n{state['draft_report']}"),
        },
    ]
    findings: list[str] = []

    # ⚠️ 关键修改 1：初始值不再是 "high"
    #    以前解析失败就直接算 high —— 于是"有风险"和"没查完"塌缩成同一个值，
    #    人工看到 high 分不清是"这份合同有雷"还是"系统没跑完"。
    #    现在用 review_status 单独表达"复核是否完成"。
    risk_level = ""
    review_status = "incomplete"

    for round_no in range(MAX_ROUNDS):
        # ⚠️ 关键修改 2：最后一轮强制收口
        #    以前每一轮都给 tools，模型可以一直查下去，直到轮数耗尽才兜底。
        #    实测过一次：用例 B 就是"跑到 20 轮还没给 RISK 行"。
        forced = round_no == MAX_ROUNDS - 1
        if forced:
            messages.append({"role": "user", "content": FINISH_HINT})

        request: dict[str, Any] = {
            "model": settings.llm_model_name,
            "messages": messages,
            "temperature": settings.llm_temperature,
        }
        if not forced:
            request["tools"] = TOOLS
        # 强制收口这一轮不传 tools —— 模型没有工具可调，只能出文本

        response = await llm_client._get_client().chat.completions.create(**request)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                }
            )
            for tc in msg.tool_calls:
                result = await call_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        if not msg.content:
            findings.append("复核模型返回了空内容，无法给出结论，已升级人工确认。")
            break

        content = msg.content
        risk_match = re.search(r"(?im)^\s*RISK\s*:\s*(high|low)\s*$", content)
        findings_text = re.sub(r"(?im)^\s*RISK\s*:\s*(?:high|low)\s*$", "", content).strip()
        if findings_text:
            findings.append(findings_text)
        messages.append({"role": "assistant", "content": content})

        if risk_match:
            risk_level = risk_match.group(1).lower()
            review_status = "complete"
        else:
            # 有内容但没有 RISK 行 —— 同样是"没给出结论"，不能当成 low
            findings.append("复核模型未按格式输出 RISK 行，无法判定风险等级，已升级人工确认。")
        break
    else:
        # 走到这里说明最后一轮强制收口都没能产出文本
        findings.append("复核达到最大轮数仍未收到结论，已升级人工确认。")

    return {
        "messages": messages,
        "findings": findings,
        "risk_level": risk_level,
        "review_status": review_status,
    }


# ══════════════════ B3 关键节点 ③：是否升级到人工 ══════════════════
#
# 返回 "gate" 或 "end"。
#
# 升级人工有两种【性质完全不同】的原因，都必须拦，但必须能区分：
#
#   review_status != "complete"  -> 复核没跑完（系统/流程问题）
#   risk_level == "high"         -> 查完了，发现高危问题（内容问题）
#   其余（complete 且 low）      -> 放行
#
# 以前这两者塌缩成同一个 risk_level="high"，
# 于是人工点进来分不清"合同有雷"还是"系统没跑完" —— 处理方式完全不同。
#
# ═══════════════════════════════════════════════════════════════


def should_escalate(state: ReviewState) -> str:
    if state.get("review_status") != "complete":
        return "gate"
    return "gate" if state["risk_level"] == "high" else "end"



async def human_gate(state: ReviewState) -> dict:
    """人工确认节点（样板）。

    注意：这个函数**只在人工确认之后才会执行**。
    因为编译时带了 interrupt_before=["human_gate"]，图会在这里暂停，
    等外部恢复后再进来 —— 所以这里不需要写"等待"逻辑，等待是框架做的。
    """
    return {"confirmed": True, "findings": ["人工确认通过"]}


def build_review_graph(checkpointer: Any = None):
    """复核图的接线（样板）。"""
    builder = StateGraph(ReviewState)

    builder.add_node("review", review_node)
    builder.add_node("human_gate", human_gate)

    builder.set_entry_point("review")
    builder.add_conditional_edges(
        "review",
        should_escalate,
        {"gate": "human_gate", "end": END},
    )
    builder.add_edge("human_gate", END)

    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_gate"])


async def run_review(
    question: str,
    draft_report: str,
    *,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> dict:
    graph = build_review_graph(checkpointer)
    initial: ReviewState = {
        "question": question,
        "draft_report": draft_report,
        "findings": [],
        "risk_level": "",
        "review_status": "incomplete",
        "confirmed": False,
        "messages": [],
    }
    config = {"configurable": {"thread_id": thread_id or "review"}}
    return await graph.ainvoke(initial, config=config)
