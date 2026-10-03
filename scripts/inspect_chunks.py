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

    # ← 统计口径：前言可能被二次切分成多【块】，但它只是 1【组】。
    # 原来直接数"没有条款号的块"，会把前言的多个分片误报成"正文块没识别到条款号"
    # （Day 3 实测：B 文档因此产生了一条假警告）。
    # 假警告比没有警告更糟 —— 它会训练你忽略警告。
    first_numbered = next((i for i, c in enumerate(chunks) if c.clause_no), len(chunks))
    preamble_blocks = first_numbered  # 开头连续的无条款号块 = 前言（可能被切分）
    orphan = [c for c in chunks[first_numbered:] if not c.clause_no]

    print(f"前言：{preamble_blocks} 块（被切分属正常）")
    print(f"正文中缺条款号：{len(orphan)} 块")

    sections = Counter(c.section for c in chunks if c.section)
    if sections:
        print(f"覆盖章节：{len(sections)} 个 -> {list(sections)[:4]}")
    else:
        # ← 区分"文档本来就没有章"和"章节识别失败"：两者输出必须不同，
        # 否则这个统计等于没做（Day 3 实测：A 文档输出 0 个，无从判断是哪一种）。
        print("覆盖章节：0 个（本文档无章节级结构，或章节识别失败 —— 需人工确认）")

    if over:
        print(f"\n❌ 超过 {limit} token 的块：{len(over)} —— 必须为 0")
        for c in over:
            print(f"   {c.location}  {c.token_count} tok")
        return 1

    print("\n✅ 无超限块")
    if args.strategy == "clause" and orphan:
        print(f"⚠️ 正文中有 {len(orphan)} 块没识别到条款号（非前言），检查正则或文档结构")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
