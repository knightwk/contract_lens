"""Day 5 验收：检索 → 组装 Prompt → 核对引用 →（可选）调用 LLM。

用法
    # 只验证检索 + Prompt 组装 —— 不需要 LLM_API_KEY
    python scripts/rag_demo.py --dry-run
    python scripts/rag_demo.py --dry-run "合同什么时候生效"

    # 完整链路 —— 需要 LLM_API_KEY
    python scripts/rag_demo.py "合同什么时候生效"

    # 指定材料条数
    python scripts/rag_demo.py --dry-run --top-k 5

为什么 --dry-run 值得单独存在
    "模型能不能引用" 这件事，先在【不给模型看】的情况下检查一遍：
    如果材料里连标识和定位都没有，那模型必然只能编。
    这一半的检查不需要 API key、不花钱、也不受模型随机性影响。
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

# Windows 下 stdout 被重定向时会退化为 GBK，↳ / ✅ 这类字符会直接抛错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.llm import llm_client  # noqa: E402
from app.core.rag import answer, build_messages, extract_markers  # noqa: E402
from app.core.retriever import retrieve  # noqa: E402

DEFAULT_QUESTION = "合同什么时候生效"

# 从回答里抓「第X条」形态的引用（人读的依据）。
# 材料标识的形状不在这里定义 —— 它在 app/core/rag.py，和生成它的 _format_context
# 放在一起。分开定义的话，改了格式而忘了改校验器，"可核对"会静默失效。
_RE_CLAUSE_REF = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--dry-run", action="store_true", help="只做检索与 Prompt 组装，不调用 LLM")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--show-prompt", action="store_true", default=True)
    args = ap.parse_args()

    print("=" * 74)
    print(f"模型: {settings.llm_model_name or '(未配置)'}    temperature: {settings.llm_temperature}")
    print(f"问题: {args.question}")
    print("=" * 74)

    # ---------- 1. 检索 ----------
    print("\n[1/4] 检索")
    hits = await retrieve(args.question, top_k=args.top_k)
    print(f"  命中 {len(hits)} 条")
    for i, h in enumerate(hits, 1):
        print(f"   {i:2}. score={h.score:7.4f}  [{h.source}]  {h.location[:40]}")
    if not hits:
        print("  ❌ 没有检索到任何材料 —— 先确认 Milvus 里有数据")
        return 1

    # ---------- 2. 组装 Prompt ----------
    print("\n[2/4] 组装 Prompt")
    messages = build_messages(args.question, hits)
    total_tokens = llm_client.count_messages_tokens(messages)
    print(f"  消息数 {len(messages)}，估算 {total_tokens} tokens")
    print(f"  上下文预算上限 {settings.rag_context_max_tokens} tokens")
    if total_tokens > settings.rag_context_max_tokens:
        print(f"  ⚠️ 超出预算 —— 需要截断材料或调低 RAG_TOP_K")

    # ---------- 3. 材料可核对性检查（不需要 LLM）----------
    print("\n[3/4] 材料可核对性")
    context = messages[-1]["content"]
    hit_ids = {h.chunk_id for h in hits}
    hit_clauses = {h.clause_no for h in hits if h.clause_no}

    unique_markers = sorted(set(extract_markers(context)))
    unknown_in_ctx = [m for m in unique_markers if m not in hit_ids]
    print(f"  材料里出现的标识      : {len(unique_markers)} 个 -> {unique_markers[:4]}")
    # 加严：不只是"有一个能对上"，而是【每条】材料都有一个能指回自己的唯一标识。
    # 这才是"引用可核对"的前提 —— 有一条没标识，模型引用它时就只能编。
    has_id = bool(unique_markers) and not unknown_in_ctx and len(unique_markers) == len(hit_ids)
    has_clause = any(c in context for c in hit_clauses)
    print(f"  每条材料都有唯一标识  : {'✅' if has_id else f'❌ {len(unique_markers)}/{len(hit_ids)} 条'}")
    print(f"  材料里看得到条款号    : {'✅' if has_clause else '❌ 模型说不清依据在哪'}")

    if args.show_prompt:
        print("\n" + "-" * 74)
        print("组装好的 Prompt（原样）")
        print("-" * 74)
        for m in messages:
            print(f"\n### role={m['role']}\n{m['content']}")

    if args.dry_run:
        ok = has_id and has_clause
        print("\n" + "=" * 74)
        print("✅ dry-run 通过：检索与 Prompt 组装都正常" if ok else "❌ 材料格式还缺东西（见上面两项）")
        print("=" * 74)
        return 0 if ok else 1

    # ---------- 4. 调用 LLM + 核对引用 ----------
    print("\n[4/4] 调用 LLM")
    result = await answer(args.question, top_k=args.top_k)
    print("\n" + "-" * 74)
    print("回答")
    print("-" * 74)
    print(result.answer)

    # 两项都核对：
    #   条款号 —— 人读的依据
    #   材料标识 —— 机器核对的锚点（这才是"引用可回溯"真正依赖的东西）
    cited_clauses = set(_RE_CLAUSE_REF.findall(result.answer))
    unknown_clauses = cited_clauses - hit_clauses

    cited_markers = extract_markers(result.answer)
    unknown_markers = [m for m in cited_markers if m not in hit_ids]

    print("\n" + "-" * 74)
    print("引用核对（约定 5：每条结论都要能指回材料）")
    print("-" * 74)
    print(f"  回答引用的条款号 ({len(cited_clauses)} 个): {sorted(cited_clauses) if cited_clauses else '(无)'}")
    print(f"  回答引用的标识   ({len(cited_markers)} 个): {cited_markers[:4]}")
    print(f"  材料真实存在的标识  ({len(hit_ids)} 个)")

    if unknown_markers:
        print(f"  ❌ 编造的标识（材料里不存在）: {unknown_markers[:5]}")
    elif not cited_markers:
        print("  ❌ 回答里没有任何材料标识 —— 模型没有按输出格式引用")
    else:
        print("  ✅ 所有标识都能在材料里找到")

    if unknown_clauses:
        print(f"  ❌ 编造的条款号（材料里不存在）: {sorted(unknown_clauses)}")
    elif cited_clauses:
        print("  ✅ 所有条款号都能在材料里找到")

    print(f"\n  tokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")
    print(f"  延迟: {result.latency_ms:.0f} ms")

    ok = not unknown_markers and not unknown_clauses and bool(cited_markers)
    print("\n" + ("✅ 通过：回答的引用全部可核对" if ok else "❌ 引用核对未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
