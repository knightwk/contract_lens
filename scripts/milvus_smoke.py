"""Milvus 验收清单：建表 -> 写入 -> 幂等 -> 三条检索路径 -> 引用可回溯。

用法（项目根目录；先 docker compose up -d 并等 healthy）
    python scripts/milvus_smoke.py
    python scripts/milvus_smoke.py data/samples/clean/某合同.pdf
    python scripts/milvus_smoke.py --recreate      # 先删集合再重建

退出码：0 = 全部通过，1 = 有检查未通过
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Windows 下 stdout 被重定向时会退化为 GBK，↳ / ✅ 这类字符会直接抛错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.embedding import get_embedding_provider  # noqa: E402
from app.core.ingest import index_document  # noqa: E402
from app.core.parser import parse_document  # noqa: E402
from app.core.vector_store import vector_store  # noqa: E402

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "test_contract.md"
)

QUERIES = [
    "合同什么时候生效",   # 语义查询 -> 稠密应命中
    "第十二条",           # 编号精确匹配 -> BM25 应命中
    "违约责任",           # 关键词 -> 两者都该命中
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--recreate", action="store_true", help="先删掉集合再重建")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    print("=" * 74)
    print(f"Milvus: {settings.milvus_uri}    集合: {settings.milvus_collection_name}")
    print("=" * 74)

    # ---------- 1. 建表 ----------
    print("\n[1/5] 建表")
    created = await vector_store.ensure_collection(recreate=args.recreate)
    info = await vector_store.describe()
    print(f"  本次{'创建' if created else '复用'}集合")
    print(f"  字段：{info.get('fields')}")
    print(f"  函数：{info.get('functions')}    <- 应包含 bm25")

    if "dense" not in (info.get("fields") or []) or "sparse" not in (info.get("fields") or []):
        print("  ❌ 缺少 dense / sparse 字段")
        return 1

    # ---------- 2. 写入 ----------
    print("\n[2/5] 写入")
    doc = parse_document(args.path)
    result = await index_document(doc)
    print(f"  文档：{result['filename']}")
    print(f"  块数：{result['chunks']}（带条款号 {result['numbered']}）写入 {result['written']}")

    # ---------- 3. 幂等 ----------
    print("\n[3/5] 幂等验证（同一份文档再写一次）")
    # ← 断言必须用 count_rows()（真实可查询行数），不能用 stats_row_count()。
    #   Milvus 的 upsert 是 delete + insert，被覆盖的旧行成为墓碑，
    #   在 compaction 之前仍被 get_collection_stats 计入。
    #   Day 4 实测：这里曾误用统计值，导致正确的 upsert 被判为 ❌（10 -> 20）。
    before = await vector_store.count_rows()
    await index_document(doc)
    after = await vector_store.count_rows()
    idempotent = before == after
    print(f"  可查询行数：{before} -> {after}")
    print(f"  {'✅ 行数未变，幂等成立' if idempotent else '❌ 行数变了 —— chunk_id 不唯一，或写的是 insert'}")

    stats_count = await vector_store.stats_row_count()
    if stats_count != after:
        print(
            f"  ℹ️ get_collection_stats 报的是 {stats_count}（多出 {stats_count - after} 条"
            f"待 compaction 的墓碑）—— 这个值不能用来做断言"
        )

    # ---------- 4. 三条检索路径 ----------
    print("\n[4/5] 检索")
    provider = get_embedding_provider()
    for query in QUERIES:
        vector = await provider.embed_query(query)
        dense = await vector_store.search_dense(vector, limit=args.limit)
        bm25 = await vector_store.search_bm25(query, limit=args.limit)
        hybrid = await vector_store.hybrid_search(query, vector, limit=args.limit)
        print(f"\n  ── query: {query}")
        for name, hits in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid)):
            top = hits[0] if hits else None
            if top:
                print(f"     {name:7} top1={top.score:9.4f}   {top.location[:34]}")
            else:
                print(f"     {name:7} (无结果)")

    # ---------- 5. 引用可回溯（约定 5）----------
    print("\n[5/5] 引用可回溯检查")
    vector = await provider.embed_query(QUERIES[0])
    hits = await vector_store.search_dense(vector, limit=args.limit)
    missing_text = [h.chunk_id for h in hits if not h.text]
    missing_loc = [h.chunk_id for h in hits if not h.clause_no and not h.section]
    print(f"  命中 {len(hits)} 条")
    print(f"  缺原文：{len(missing_text)}    缺定位：{len(missing_loc)}")
    if hits:
        h = hits[0]
        print(f"  示例 chunk_id={h.chunk_id}")
        print(f"       location={h.location}")
        print(f"       page={h.page}  char=[{h.char_start}, {h.char_end})")
        print(f"       原文：{h.text[:60]}")
    ok_cite = not missing_text

    # ---------- 汇总 ----------
    final_info = await vector_store.describe()
    real_rows = await vector_store.count_rows()
    print("\n" + "=" * 74)
    print(f"集合 {final_info['collection']}：可查询 {real_rows} 行（统计值 {final_info['row_count']}）")
    print("=" * 74)

    checks = {
        "bm25 函数存在": "bm25" in (info.get("functions") or []),
        "写入幂等": idempotent,
        "引用带原文": ok_cite,
    }
    print()
    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name}")

    ok = all(checks.values())
    print("\n" + ("✅ 全部通过" if ok else "❌ 有检查未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
