"""Day 2 冒烟测试：验证本地模型可用，并记录后续优化要用的基线数字。

用法（项目根目录）：
    python scripts/smoke_test_models.py

离线验证（另开终端，先设环境变量）：
    $env:HF_HUB_OFFLINE="1"; python scripts/smoke_test_models.py
"""

import asyncio
import sys
import time
from pathlib import Path

# 直接跑脚本时，Python 只把 scripts/ 加进 sys.path，import app 会失败
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.model_runtime import model_runtime  # noqa: E402

SIM_A = "本合同自双方签字之日起生效"
SIM_B = "本协议自双方签署之日起生效"  # 与 A 语义相似
SIM_C = "乙方应在三十日内支付全部货款"  # 与 A 无关

QUERY = "合同什么时候生效"
RERANK_DOCS = [
    "本合同自双方签字之日起生效",  # 相关
    "乙方应在三十日内支付全部货款",  # 无关
    "本协议经双方盖章后生效",  # 相关
]

results: list[tuple[str, str, str]] = []  # (项目, 结果, 判定)


def record(name: str, value: str, ok: bool | None = None) -> None:
    verdict = "—" if ok is None else ("PASS" if ok else "FAIL")
    results.append((name, value, verdict))


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


async def main() -> int:
    print("=" * 68)
    print("ContractLens 本地模型冒烟测试")
    print("=" * 68)
    print(f"embedding : {settings.embedding_model_name}")
    print(f"            {settings.embedding_model_path}")
    print(f"rerank    : {settings.rerank_model_name}")
    print(f"            {settings.rerank_model_path}")
    print(f"torch 线程 : {settings.torch_num_threads}   并发上限: {settings.max_concurrent_inference}")
    print()

    # ---------- 1. 冷启动 ----------
    print("[1/5] 冷启动耗时")
    t0 = time.perf_counter()
    await model_runtime.warmup_embedder()
    cold_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    await model_runtime.warmup_reranker()
    cold_rerank = time.perf_counter() - t0

    record("冷启动 embedding", f"{cold_embed:.2f}s")
    record("冷启动 rerank", f"{cold_rerank:.2f}s")
    print()

    # ---------- 2. 相似度 ----------
    print("[2/5] 语义相似度")
    vectors = await model_runtime.encode_async([SIM_A, SIM_B, SIM_C], is_query=False)
    sim_related = cosine(vectors[0], vectors[1])
    sim_unrelated = cosine(vectors[0], vectors[2])
    print(f"  cos(生效, 签署)  = {sim_related:.4f}   ← 相似")
    print(f"  cos(生效, 付款)  = {sim_unrelated:.4f}   ← 无关")
    record("相似 > 无关", f"{sim_related:.4f} > {sim_unrelated:.4f}", sim_related > sim_unrelated)
    print()

    # ---------- 3. 精排 ----------
    print("[3/5] 精排排序")
    scores = await model_runtime.rerank_async(QUERY, RERANK_DOCS)
    for doc, score in zip(RERANK_DOCS, scores):
        print(f"  {score:>10.4f}  {doc}")
    print(f"  分数范围: {min(scores):.4f} ~ {max(scores):.4f}   ← 若为 0~1 说明已套 Sigmoid；若含负数则为原始 logits")
    # 相关文档应排在无关文档之前
    best_irrelevant = scores[1]
    ok_rerank = scores[0] > best_irrelevant and scores[2] > best_irrelevant
    record("相关文档排在无关之前", f"max(相关)={max(scores[0], scores[2]):.4f} > 无关={best_irrelevant:.4f}", ok_rerank)
    print()

    # ---------- 4. query 指令前缀 A/B ----------
    print("[4/5] query 指令前缀影响")
    original = settings.embedding_query_instruction
    try:
        settings.embedding_query_instruction = ""
        no_prefix = (await model_runtime.encode_async([QUERY], is_query=True))[0]
        settings.embedding_query_instruction = original
        with_prefix = (await model_runtime.encode_async([QUERY], is_query=True))[0]
    finally:
        settings.embedding_query_instruction = original

    doc_vec = (await model_runtime.encode_async([SIM_A], is_query=False))[0]
    cos_no = cosine(no_prefix, doc_vec)
    cos_yes = cosine(with_prefix, doc_vec)
    print(f"  不加前缀 cos(query, doc) = {cos_no:.4f}")
    print(f"  加前缀   cos(query, doc) = {cos_yes:.4f}")
    print(f"  差异 = {cos_yes - cos_no:+.4f}   ← 决定 W2 是否保留前缀")
    record("前缀增益", f"{cos_yes - cos_no:+.4f}", cos_yes > cos_no)
    print()

    # ---------- 5. 延迟基线 ----------
    print("[5/5] 延迟基线")
    t0 = time.perf_counter()
    await model_runtime.encode_async([SIM_A], is_query=False)
    dt_single = time.perf_counter() - t0

    batch = [f"第 {i} 条：本合同项下的义务应当在约定期限内履行。" for i in range(32)]
    t0 = time.perf_counter()
    await model_runtime.encode_async(batch, is_query=False)
    dt_batch = time.perf_counter() - t0

    ten_docs = RERANK_DOCS * 4  # 12 条，近似 Top-10 的量级
    ten_docs = ten_docs[:10]
    t0 = time.perf_counter()
    await model_runtime.rerank_async(QUERY, ten_docs)
    dt_rerank10 = time.perf_counter() - t0

    record("encode 单条", f"{dt_single * 1000:.1f} ms")
    record("encode 批量 32", f"{dt_batch * 1000:.1f} ms  ({dt_batch / 32 * 1000:.1f} ms/条)")
    record("rerank Top-10", f"{dt_rerank10 * 1000:.1f} ms   ← W2 优化的起点")
    print()

    # ---------- 汇总 ----------
    print("=" * 68)
    print(f"{'项目':<24}{'数值':<34}{'判定'}")
    print("-" * 68)
    for name, value, verdict in results:
        print(f"{name:<24}{value:<34}{verdict}")
    print("=" * 68)

    failed = [r for r in results if r[2] == "FAIL"]
    if failed:
        print(f"\n❌ {len(failed)} 项未通过")
        return 1
    print("\n✅ 全部通过。请把上面的数字记下来——W2 和 W4 要拿它们当基线。")
    print("   还差一步：设 HF_HUB_OFFLINE=1 重跑一次，确认模型目录自包含。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
