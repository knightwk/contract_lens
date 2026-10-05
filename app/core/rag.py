"""最小 RAG 链路：检索 → 组装 Prompt → 调 LLM → 带引用的回答。

为什么单独成文件
    retriever 只管"拿材料"，llm 只管"调模型"，
    「把材料组织成让模型能引用的形式」是第三种职责 —— 而且是最容易出错的一种：
    材料给得不清楚，模型就只能编。

Day 5 的范围
    流式 UI、结构化输出、风险识别都是后面周次的事。
    这里只做一件事：**让回答基于原文，并且每条结论都能指回出处**。
"""

import re

from app.core.llm import LLMError, llm_client
from app.core.logger import logger
from app.core.retriever import retrieve
from app.core.schemas import RagAnswer, SearchHit

# 系统提示词。两条硬要求都来自 docs/conventions.md 约定 5：
#   ① 不得编造 —— 材料里没有的必须明说，而不是猜一个像样的答案
#   ② 必须引用 —— 每条结论标出它来自哪一条材料
#
# 「输出格式」里 [<材料标识>] 的写法必须与 _format_context 生成的标识【逐字一致】——
# 否则模型不知道该怎么引用，你也无从核对。
#
# ⚠️ 低温管的是【采样随机性】，不管【编造】。幻觉是指令遵循问题，不是随机性问题，
#    所以约束不会因为 temperature=0.1 就可以省掉。
SYSTEM_PROMPT = """你是一名资深的合同审查专员，
根据【材料】回答【问题】。
约束：
    1. 只能依据材料回答，材料中没有相关内容就说“材料中无相关内容，无法回答。”
    2. 每条结论必须标出来自哪一条材料。
输出格式：
先给结论，再列依据。

结论：<一句话回答>
依据：
- <条款号> <条款标题> · 第 <页码> 页 · [<材料标识>]
- ……
"""


# ── 材料标识 ────────────────────────────────────────────────
# 这个正则必须与 _format_context 生成的标识形状【保持同步】。
# 故意放在一起：改了材料格式就必须改这里，否则"引用可核对"会静默失效 ——
# 校验器抓不到标识时，它不会报错，只会告诉你"没有引用"。
MATERIAL_MARKER_RE = re.compile(r"\[([0-9a-f]{8,})\]")


def extract_markers(text: str) -> list[str]:
    """从一段文本里抓出所有材料标识。

    用于核对：模型回答里的引用是不是真的来自材料。
    """
    return MATERIAL_MARKER_RE.findall(text)


def _format_context(hits: list[SearchHit]) -> str:
    """把检索结果排版成提示词里的【材料】部分。

    每条材料 = [标识] + 定位信息 + 换行 + 正文；材料之间用空行分隔。

    两个设计点：
      · **标识独立成一个方括号** —— 它是模型要【精确复现】的东西，
        和给人读的定位信息不是一回事。混在同一个括号里的话，模型引用时
        得把章节、条款、标题、页码全抄一遍，抄错一个字就核对不上。
      · **标识用 chunk_id 而不是 clause_no** —— clause_no 会重复
        （附件里也有"第一条"；实测那份养老合同里"第一条"出现了两次）。
    """
    context = "\n\n".join(
        f"[{hit.chunk_id}] 来源：{hit.section or '(无章节)'} · {hit.clause_no or ''} · {hit.clause_title or ''} 页码：{hit.page or '无页码'}\n{hit.text}"
        for hit in hits
    )
    return context


def build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    """组装送给 LLM 的完整消息列表：系统提示词 + 一段含材料与问题的用户消息。"""
    context = _format_context(hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【材料】\n\n{context}\n\n【问题】\n\n{question}",
        },
    ]


async def answer(question: str, *, top_k: int | None = None) -> RagAnswer:
    """完整链路：检索 → 组装 → 调用 → 返回带材料的回答。

    注意返回的是 RagAnswer，它把 hits 一起带回来了 ——
    这样调用方能核对"模型引用的那条真的存在吗"（约定 5）。
    """
    hits = await retrieve(question, top_k=top_k)
    if not hits:
        logger.warning(f"没有检索到任何材料：{question}")
        return RagAnswer(question=question, answer="没有检索到相关条款。")

    messages = build_messages(question, hits)

    # 上下文预算：材料太多会撑爆模型窗口，也会让成本失控
    estimated = llm_client.count_messages_tokens(messages)
    logger.info(f"检索 {len(hits)} 条材料，提示词约 {estimated} tokens")

    try:
        result = await llm_client.complete(messages)
    except LLMError as exc:
        logger.error(f"LLM 调用失败：{exc}")
        raise

    return RagAnswer(
        question=question,
        answer=result.text,
        hits=hits,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
    )
