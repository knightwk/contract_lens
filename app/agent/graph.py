"""LangGraph 版本的 Agent 循环。

为什么要再写一遍
    手写版你已经踩过四个坑：tool_call_id 对应、预算要倒数、新增信息要先算差集、
    schema 键名拼错静默失效。
    现在用框架写同一件事 —— 目的不是"能跑"，而是回答那个面试必问的问题：

        **LangGraph 到底帮你做了什么？**

────────────────────────────────────────────────────────────────
本文件含 3 个 TODO（Agent 节点的三个函数）。
图的接线我已经写好了 —— 因为那是样板，不构成判断。
────────────────────────────────────────────────────────────────
"""

import json
import operator
import sys
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

sys.path.append(r"F:\llm\projects\contract_lens")
from app.agent.tools import TOOLS, call_tool
from app.config import settings
from app.core.llm import llm_client
from app.core.rag import MATERIAL_MARKER_RE

MAX_STEPS = 6

SYSTEM_PROMPT = """你是一个合同审查助手。

你可以调用工具来查合同条款。规则：
1. 需要合同里的具体内容时，先用工具查，不要凭印象回答
2. 可以多次调用工具，但不要重复查同样的内容
3. 每一条结论都要引用材料标识，格式：[<标识>]
4. 材料里没有的，就明说做不到 —— 不要猜
# 注意：你还能调用 {N} 次工具。
"""


# ══════════════════ 新概念：Reducer ══════════════════
#
# 这是手写版【没有】的东西，也是你最该在面试里讲的点。
#
# 手写版里，状态就是一个普通变量：
#     chunk_ids.append(...)        # 你想加就加，想改就改
#     steps += 1
# 加和改没有区别 —— 全靠你记得别写错。
#
# LangGraph 里，**每个字段的更新方式必须显式声明**：
#
#     Annotated[list, operator.add]     -> 节点返回的值【追加】到已有值后面
#     不加 Annotated                    -> 节点返回的值【覆盖】已有值
#
# 为什么必须有这个区分
#     图是多节点的，每个节点只返回"我这一轮的增量"。
#     如果 `steps` 也用追加，你会得到一个 [0,1,1,2,3] 这种看不懂的列表；
#     如果 `chunk_ids` 用了覆盖，第二个节点一返回就把前面累积的全冲掉了。
#
# 🔴 最容易犯的错
#     该追加的字段忘了加 Annotated -> 它在每轮被覆盖 -> **永远只有一个元素**。
#     而且不会报错，只是行为不对 —— 又是一次静默失效。


class AgentState(TypedDict):
    """图的状态。

    判断每个字段该用"追加"还是"覆盖"：
        这一轮结束后，已有的值还需要保留吗？
    """

    question: str  # 覆盖（只写一次，不变）
    # 消息必须追加：每一轮的 assistant / tool 消息都要保留，否则模型看不到历史
    messages: Annotated[list, operator.add]
    # 已见过的条目必须追加：它记录【跨轮次】的累积
    chunk_ids: Annotated[list[str], operator.add]
    # 预算轨迹必须追加：用来观测倒数是否生效
    budget_trace: Annotated[list[int], operator.add]
    steps: int  # 覆盖（每轮 +1，只要当前值）


# ══════════════════ TODO-A4-1：Agent 节点 ══════════════════


async def agent_node(state: AgentState) -> dict:
    remaining = MAX_STEPS - state["steps"] - 1
    # state["messages"][0]["content"] = SYSTEM_PROMPT
    state["messages"][0]["content"] = SYSTEM_PROMPT.format(N=remaining)
    response = await llm_client._get_client().chat.completions.create(
        model=settings.llm_model_name,
        messages=state["messages"],
        temperature=settings.llm_temperature,
        tools=TOOLS,
    )

    msg = response.choices[0].message
    res = {
        "steps": state["steps"] + 1,
        "budget_trace": [remaining],
        "messages": [
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else [],
            }
        ],
    }
    return res


#
# 要做的事：
#   ① 算剩余步数并注入 system 提示：remaining = MAX_STEPS - state["steps"]
#   ② 用完整消息历史调模型（带 tools）
#   ③ 返回【增量】而不是完整消息列表 —— 因为 messages 是追加语义，
#      你只需要返回这一轮新出现的消息
#
# 注意两件事：
#   · state["steps"] 是【进入本节点时】的值，不是 +1 之后的
#   · 返回值里如果没有某个键，那个键的状态就不变
#
# ═══════════════════════════════════════════════════════════════


# ══════════════════ TODO-A4-2：工具节点 ══════════════════
async def tools_node(state: AgentState) -> dict:
    tools_msgs = []
    chunk_ids = []
    hint = []
    exists_chunk_ids = state["chunk_ids"]
    tool_calls = state["messages"][-1]["tool_calls"]

    for tc in tool_calls:
        result = await call_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        tools_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        if tc["function"]["name"] in ("search_clauses", "get_clause_details"):
            this_chunk_ids = set(MATERIAL_MARKER_RE.findall(result))
            new_chunk_ids = this_chunk_ids - set(exists_chunk_ids)
            if not new_chunk_ids:
                hint = [{"role": "user", "content": "本次检索没有新增信息，当前信息应该可以回答了。"}]
            chunk_ids.extend(new_chunk_ids)

    res = {"messages": tools_msgs + hint, "chunk_ids": chunk_ids, "steps": state["steps"]}
    return res


# 要做的事：
#   ① 从 state["messages"] 里最后一条 assistant 消息取出 tool_calls
#   ② 逐个执行，把结果作为 role="tool" 的消息收集起来
#   ③ 只有 search_clauses / get_clause_details 才做新增信息检测
#   ④ 返回：新增的 tool 消息 + 本轮新见到的 chunk_id + 本轮剩余步数
#
# 🔑 这里有个和手写版【不一样】的地方，值得注意：
#
#   手写版里你必须「先算差集、再累积」，顺序写反就永远算不出新增。
#   在 LangGraph 里，因为 chunk_ids 是追加语义，
#   你只需要 **返回本轮新增的 id** —— 累积由 reducer 负责。
#
#   也就是说：**框架把"读状态"和"写增量"分开了，那个顺序陷阱消失了。**
#   这是"框架帮你做了什么"的一个具体答案，面试可以直接用。
#
# ═══════════════════════════════════════════════════════════════


# ══════════════════ TODO-A4-3：条件边 ══════════════════
def should_continue(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
        return "end"

    msg = state["messages"][-1]
    if msg.get("tool_calls"):
        return "tools"
    elif msg.get("content"):
        return "end"
    else:
        return "end"


# 返回 "tools" 或 "end"。要判断三种情况：
#   ① 最后一条消息有 tool_calls        -> "tools"
#   ② 没有 tool_calls，但有内容        -> "end"（模型给答案了）
#   ③ 步数用尽                        -> "end"
#
# 返回值必须是 add_conditional_edges 里映射表出现过的键。
#
# ═══════════════════════════════════════════════════════════════


def build_graph():
    """接线（样板，已写好）。

    对比手写版的 for 循环：
        手写：   for _ in range(max_steps): ...
        图：     节点 + 边 + 条件边，循环由「边」表达

    条件边就是手写版里的 if/break —— 只不过它被声明成一张"路由表"。
    """
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()


async def run_graph(question: str, *, max_steps: int = MAX_STEPS) -> dict:
    """跑一次图，返回最终状态。"""
    graph = build_graph()
    initial: AgentState = {
        "question": question,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(N=max_steps)},
            {"role": "user", "content": question},
        ],
        "chunk_ids": [],
        "budget_trace": [],
        "steps": 0,
    }
    return await graph.ainvoke(initial)
