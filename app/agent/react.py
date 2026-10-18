"""手写的 ReAct 循环 —— 不借助任何框架。

为什么先手写
    LangGraph / CrewAI 这些框架帮你做的事，本质就是这个循环加上状态管理和断点续跑。
    手写过一遍，你才知道框架替你做了什么 —— 面试问「LangGraph 帮你解决了什么」
    时，只会调 API 的人答不出来。

    手写版本还有一个用途：后面和 LangGraph 版本做对比，
    那份对比记录本身就是面试素材（对应 JD2「理解框架源码者加分」）。

ReAct = Reason + Act
    模型先想（要不要调工具、调哪个），再行动（调工具拿结果），
    把结果作为新的观察喂回去，再想……直到它认为可以回答。

────────────────────────────────────────────────────────────────
本文件含 1 个 TODO（A1 的关键节点，你要自己写）。
────────────────────────────────────────────────────────────────
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import TOOLS, call_tool
from app.config import settings
from app.core.llm import llm_client

SYSTEM_PROMPT = """你是一个合同审查助手。

你可以调用工具来查合同条款。规则：
1. 需要合同里的具体内容时，先用工具查，不要凭印象回答
2. 可以多次调用工具，但不要重复查同样的内容
3. 每一条结论都要引用材料标识，格式：[<标识>]
4. 材料里没有的，就明说做不到 —— 不要猜
"""

MAX_STEPS = 6


@dataclass
class AgentRun:
    """一次 Agent 运行的完整记录，用于调试和成本统计。"""

    question: str
    answer: str = ""
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 步数用尽仍未得出结论。
    # 为什么需要这个布尔值：兜底文案也是非空字符串，光看 answer 分不出
    # "答出来了"和"没答出来" —— 这就是 D-027 那类假阳性。
    # **凡是"下游要判断成败"的地方，都必须给一个机器可读的信号。**
    budget_exhausted: bool = False

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tool_calls": self.tool_call_count,
            "budget_exhausted": self.budget_exhausted,
            "tools_used": [c.get("function", {}).get("name", "?") for c in self.tool_calls],
        }


# ══════════════════ A1 关键节点：手写 ReAct 循环 ══════════════════
#
# 要实现 run_agent(question, max_steps=MAX_STEPS) -> AgentRun。
#
# 一、消息协议（这些事实给你，不用猜）
#
#   1. 请求带工具：
#        response = await llm_client._get_client().chat.completions.create(
#            model=..., messages=messages, tools=TOOLS, temperature=0.1,
#        )
#      或直接复用 llm_client，但要注意 complete() 没有透传 tools ——
#      这一步先用 AsyncOpenAI 直调，后面可以给 complete() 加参数。
#
#   2. 模型想调工具时，长得是：
#        msg = response.choices[0].message
#        msg.tool_calls      -> list，每个有 .id / .function.name / .function.arguments(JSON 字符串)
#        msg.content         -> 通常是 None
#
#   3. 模型给出最终答案时：
#        msg.tool_calls is None 或 空列表，msg.content 有内容
#
#   4. 🔴 最容易出错的地方（面试也爱问）：
#      历史里必须**原样**放回 assistant 那条带 tool_calls 的消息，
#      然后再为**每一个** tool_call 追加一条 {"role": "tool", ...}，且
#      `tool_call_id` 必须和 assistant 的那条一一对应。
#      少一条、或 id 对不上，API 会直接报错。
#
#          messages.append({
#              "role": "assistant",
#              "content": msg.content or "",
#              "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
#          })
#          for tc in msg.tool_calls:
#              result = await call_tool(tc.function.name, json.loads(tc.function.arguments))
#              messages.append({
#                  "role": "tool",
#                  "tool_call_id": tc.id,
#                  "content": result,
#              })
#
# 二、你要决定的三件事（这才是关键节点）
#
#   ① 循环什么时候停？
#      至少三种情况要想：模型不再调工具 / 超过 max_steps / 模型输出异常。
#      超限时**返回什么**？直接抛错，还是返回"我没能在 N 步内完成"？
#      返回"我没能在 N 步内完成"，借此来明确告诉用户为什么没完成，并不是服务有问题。
#
#   ② 每一轮的 assistant 消息都要进历史吗？
#      不进会怎样？进了但漏了 tool_calls 会怎样？
#      应该都需要进去。
#
#   ③ max_steps 应该多大？
#      太小 -> 一个问题查不完；太大 -> 一次运行烧掉很多 token。
#      先给个初值，跑了实验再调 —— **记录你观察到的实际步数**。
#
# 三、验收
#
#   python scripts/agent_demo.py "合同什么时候生效"
#     -> 应该看到：第 1 步调了 search_clauses，第 2 步给出带 [标识] 的答案
#     -> AgentRun.summary() 里 steps=2, tool_calls=1
#
#   再试一个需要多步的："这份合同有哪些违约责任条款，分别是第几条？"
#     -> 应该看到多次工具调用
#
# ══════════════════════════════════════════════════════════════════


async def run_agent(question: str, *, max_steps: int = MAX_STEPS) -> AgentRun:
    """跑一个 ReAct 循环，返回完整运行记录。"""
    steps = 0
    answer = ""
    tool_calls = []
    budget_exhasuted = False
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_steps):
        steps += 1
        response = await llm_client._get_client().chat.completions.create(
            model=settings.llm_model_name,
            messages=messages,
            tools=TOOLS,
            temperature=settings.llm_temperature,
        )

        msg = response.choices[0].message
        if msg.tool_calls:
            this_tool_calls = [tc.model_dump() for tc in msg.tool_calls]
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": this_tool_calls,
                }
            )
            tool_calls.extend(this_tool_calls)
            for tc in msg.tool_calls:
                result = await call_tool(tc.function.name, json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        elif msg.content:
            answer = msg.content
            break

        else:
            break
    else:
        answer = f"我没能在 {max_steps} 步内完成。"
        budget_exhasuted = True

    return AgentRun(
        question=question,
        answer=answer,
        steps=steps,
        tool_calls=tool_calls,
        messages=messages,
        budget_exhausted=budget_exhasuted,
    )
