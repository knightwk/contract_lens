"""打印切分结果 + 统计。计划文档说"打印出来看过一遍（不能省）"——这就是那个脚本。

用法（项目根目录）：
    python scripts/inspect_chunks.py tests/fixtures/test_contract.md
    python scripts/inspect_chunks.py data/samples/xxx.pdf --strategy recursive
    python scripts/inspect_chunks.py tests/fixtures/test_contract.md --full
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# ← Windows 下 Python 对【真实控制台】走 Unicode 通道，但 stdout 一旦被重定向
# （管道 / 文件 / CI），就会退化为本地编码 cp936(GBK)，本脚本用到的 ↳ 和 ⏎
# 会直接抛 UnicodeEncodeError，而不是打印出来。
# 这个 bug 只在重定向时出现 —— 恰恰是最不容易被发现的地方（Day 3 实测踩到）。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.chunker import chunk_document, max_tokens
from app.core.parser import DocumentParseError, parse_document


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="文档路径（.pdf / .docx / .md / .txt）")
    ap.add_argument("--strategy", default="clause", choices=["clause", "recursive", "length"])
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="打印每个块的完整文本")
    args = ap.parse_args()

    try:
        doc = parse_document(args.path)
    except DocumentParseError as exc:
        print(f"❌ 解析失败：{exc}")
        return 1

    kwargs = {}
    if args.max_chars:
        kwargs["max_chars"] = args.max_chars
        kwargs["size"] = args.max_chars  # 兼容三种策略的参数名

    chunks = chunk_document(doc, strategy=args.strategy, **kwargs)

    limit = max_tokens()
    over = [c for c in chunks if c.token_count > limit]

    print("=" * 78)
    print(f"文档：{doc.filename}   策略：{args.strategy}   块数：{len(chunks)}")
    print(f"模型上限：{limit} tokens   切分上限：{settings.chunk_max_chars} 字")
    print("=" * 78)

    for i, c in enumerate(chunks, 1):
        tag = "  ↳" if c.is_split else "   "
        loc = c.location[:30]
        text = c.text.replace("\n", " ⏎ ")
        shown = text if args.full else (text[:46] + ("…" if len(text) > 46 else ""))
        tq = "  ⚠️超限" if c.token_count > limit else ""
        print(f"[{i:02d}]{tag} {loc:<30} │ {len(c.text):>4} 字 │ {c.token_count:>4} tok{tq}")
        print(f"        {shown}")

    print("-" * 78)
    lengths = [len(c.text) for c in chunks]
    print(f"块数 {len(chunks)} | 平均 {sum(lengths) // len(lengths)} 字 | 最长 {max(lengths)} 字")

    numbered = [c for c in chunks if c.clause_no]
    unnumbered = [c for c in chunks if not c.clause_no]

    print(f"带条款号：{len(numbered)} 块    |    无条款号：{len(unnumbered)} 块（前言 / 附件 / 签署区）")

    sections = Counter(c.section for c in chunks if c.section)
    if sections:
        print(f"覆盖章节：{len(sections)} 个 -> {list(sections)[:4]}")
    else:
        # ← 区分"文档本来就没有章"和"章节识别失败"：两者输出必须不同，
        # 否则这个统计等于没做（Day 3 实测：A 文档输出 0 个，无从判断是哪一种）。
        print("覆盖章节：0 个（本文档无章节级结构，或章节识别失败 —— 需人工确认）")

    # ← P1 的核心可观测指标：单个条款吞了多少块。
    # 修复前 A 文档「第十五条」占 33/78 = 42%（附件和签署页全被算进最后一条）。
    # 修复后应显著下降；如果仍然很高，说明还有哨兵形态没覆盖到。
    groups = Counter(c.clause_no for c in numbered)
    if groups:
        top_no, top_n = groups.most_common(1)[0]
        share = top_n / len(chunks)
        flag = "   ⚠️ 单条款占比 > 25%，可能仍在吞附件" if share > 0.25 else ""
        print(f"条款组：{len(groups)} 个    最大：{top_no} -> {top_n} 块（{share:.0%}）{flag}")

    if over:
        print(f"\n❌ 超过 {limit} token 的块：{len(over)} —— 必须为 0")
        for c in over:
            print(f"   {c.location}  {c.token_count} tok")
        return 1

    print("\n✅ 无超限块")
    if args.strategy == "clause" and not numbered:
        # 注意：无条款号的块现在包含【附件与签署区】，是正常形态，不再据此告警。
        # 只有「一个条款号都没识别到」才说明条款正则失效。
        print("⚠️ 没有任何块识别到条款号 —— 检查正则，或该文档确实没有条款结构")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
