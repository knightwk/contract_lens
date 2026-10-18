"""Day 6 验收：跑通上传与问答两个接口。

用法
    python scripts/api_smoke.py            # 正常路径
    python scripts/api_smoke.py --errors   # 额外跑失败路径

为什么用 TestClient 而不是起 uvicorn 再 curl
    TestClient 会【完整触发 lifespan】—— 照样加载并预热模型，
    走的是和真实服务一样的代码路径；但不用管端口、不用起子进程、不用清进程。
    手工 curl 的等价命令在结尾会打印出来。

为什么这里要检查引用的【字段完整性】
    引用缺字段这种事，等前端联调时才发现就太晚了 ——
    契约变更的重灾区就是这个模型，所以单独设一项检查。
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "test_contract.md"
QUESTION = "合同什么时候生效"

# 约定 5：引用必须能让用户核对。这三样缺一不可。
REQUIRED_CITATION_KEYS = {
    "chunk_id": "可回溯的知识库条目 ID",
    "original_text": "被引用的原文",
    "clause_location": "人类可读的定位",
}


def check_citation_shape(citations: list[dict]) -> bool:
    print("\n[3/5] 引用字段完整性")
    if not citations:
        print("  ❌ 响应里没有 citations —— 回答没有带依据")
        return False
    sample = citations[0]
    print(f"  字段: {sorted(sample.keys())}")
    ok = True
    for key, why in REQUIRED_CITATION_KEYS.items():
        has = key in sample
        ok = ok and has
        print(f"  {'✅' if has else '❌'} {key:18} —— {why}")
    if not ok:
        print("  👉 见 app/api/v1/schemas.py 的 Citation 契约")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true", help="额外跑失败路径")
    args = ap.parse_args()

    results: dict[str, bool] = {}

    with TestClient(app) as client:
        # ---------- 1. 健康检查 ----------
        print("\n[1/5] GET /health")
        r = client.get("/health")
        print(f"  {r.status_code}  models_ready={r.json().get('models_ready')}")
        results["健康检查"] = r.status_code == 200

        # ---------- 2. 上传 ----------
        print("\n[2/5] POST /api/v1/documents/upload")
        with FIXTURE.open("rb") as fh:
            r = client.post(
                "/api/v1/documents/upload",
                files={"file": (FIXTURE.name, fh, "text/markdown")},
            )
        print(f"  {r.status_code}  {r.text[:200]}")
        results["上传"] = r.status_code == 200
        first = r.json() if r.status_code == 200 else {}

        # ---------- 3. 问答 ----------
        print("\n[3/5] POST /api/v1/query")
        r = client.post("/api/v1/query", json={"question": QUESTION})
        print(f"  {r.status_code}")
        if r.status_code == 200:
            body = r.json()
            print(f"  回答: {body.get('answer', '')[:120]}")
            print(f"  timings: {body.get('timings')}")
            results["问答"] = bool(body.get("answer"))
            results["引用完整性"] = check_citation_shape(body.get("citations", []))
        else:
            print(f"  {r.text[:300]}")
            results["问答"] = False
            results["引用完整性"] = False

        # ---------- 4. 重复上传 ----------
        print("\n[4/5] 重复上传同一份合同")
        with FIXTURE.open("rb") as fh:
            r2 = client.post(
                "/api/v1/documents/upload",
                files={"file": (FIXTURE.name, fh, "text/markdown")},
            )
        if r2.status_code == 200 and first:
            second = r2.json()
            print(f"  doc_id  相同？ {first.get('doc_id') == second.get('doc_id')}")
            print(f"  第一次 written={first.get('written')}  第二次 written={second.get('written')}")
            print("  👉 幂等性由确定性 chunk_id + upsert 保证（见 decisions.md D-001 / D-020）")
        results["重复上传可重入"] = r2.status_code == 200

        # ---------- 5. 失败路径 ----------
        if args.errors:
            print("\n[5/5] 失败路径")
            cases = [
                ("空问题", client.post("/api/v1/query", json={"question": ""})),
                ("超长问题", client.post("/api/v1/query", json={"question": "x" * 600})),
                ("越界 top_k", client.post("/api/v1/query", json={"question": QUESTION, "top_k": 999})),
                (
                    "不支持的类型",
                    client.post(
                        "/api/v1/documents/upload",
                        files={"file": ("a.txt", b"hello", "text/plain")},
                    ),
                ),
                ("空文件", client.post("/api/v1/documents/upload", files={"file": ("a.md", b"", "text/markdown")})),
                ("知识库里没有的问题", client.post("/api/v1/query", json={"question": "月球上的租金是多少"})),
            ]
            for name, resp in cases:
                print(f"  {name:16} -> {resp.status_code}  {resp.text[:90]}")
        else:
            print("\n[5/5] 失败路径（--errors 才跑）")

    # ---------- 汇总 ----------
    print("\n" + "=" * 74)
    for name, passed in results.items():
        print(f"  {'✅' if passed else '❌'} {name}")
    ok = all(results.values())
    print("=" * 74)
    print("\n手工 curl 的等价命令（先 uvicorn app.main:app --reload）：")
    print(f'  curl http://127.0.0.1:8000/health')
    print(f'  curl -F "file=@{FIXTURE}" http://127.0.0.1:8000/api/v1/documents/upload')
    print(f'  curl -X POST http://127.0.0.1:8000/api/v1/query \\')
    print(f'       -H "Content-Type: application/json" -d \'{{"question": "{QUESTION}"}}\'')

    print("\n" + ("✅ 全部通过" if ok else "❌ 有检查未通过"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
