"""验证 asyncio.to_thread 到底解决了什么 —— 用【并发】而不是单请求来测。

为什么单请求测不出来
    to_thread 不改变被调用函数的耗时：同一个函数、同一份 CPU、还有 GIL。
    所以"跑一次上传，前后对比秒数"必然得出"没有区别"。

    它改变的是【别的请求要不要等】。所以正确的指标是：
    跑一个重活的同时，一个轻任务要多久才能被调度。

测什么
    两件事同时开始：
      heavy() —— 解析一份真实 PDF（几百毫秒的 CPU 密集工作）
      light() —— 什么也不做，只记录"我什么时候开始跑"

    light 开始得越晚，说明事件循环被 heavy 卡得越久。

预期
    ❌ 不丢线程池：light 要等 heavy 整个跑完才轮得上 -> 几百毫秒
    ✅ 丢线程池  ：light 几乎立刻被调度            -> 不到 1 毫秒
    heavy 本身的耗时两者【几乎一样】—— 这正是要证明的点。
"""

import asyncio
import sys
from pathlib import Path
from time import perf_counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.parser import parse_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pick_heavy_file() -> Path | None:
    """优先用真实 PDF（几百毫秒）；没有就退回 fixture（太轻，可能测不出差距）。"""
    for pattern in ("data/samples/clean/*.pdf", "tests/fixtures/*.md"):
        found = sorted(ROOT.glob(pattern))
        if found:
            return found[0]
    return None


async def measure(path: Path, *, use_thread: bool) -> tuple[float, float]:
    """返回 (轻任务被调度的延迟 ms, 重任务总耗时 ms)。"""
    t_start = perf_counter()
    light_at: float | None = None

    async def heavy() -> None:
        if use_thread:
            await asyncio.to_thread(parse_document, str(path))
        else:
            parse_document(str(path))  # ← 阻塞事件循环

    async def light() -> None:
        nonlocal light_at
        # 被调到的那一刻就记时间戳。自己不做任何事，只量"轮到我"的延迟。
        light_at = (perf_counter() - t_start) * 1000

    # gather 会按顺序把它们排进事件循环：heavy 先跑
    await asyncio.gather(heavy(), light())
    heavy_ms = (perf_counter() - t_start) * 1000
    return (light_at if light_at is not None else -1.0), heavy_ms


async def main() -> int:
    target = pick_heavy_file()
    if target is None:
        print("没有可用的样本文件")
        return 1

    size_kb = target.stat().st_size / 1024
    print("=" * 74)
    print(f"样本: {target.name}  ({size_kb:.0f} KB)")
    print("=" * 74)

    # 各跑两轮，取消第一次的缓存/预热影响
    for _ in range(2):
        await measure(target, use_thread=False)
        await measure(target, use_thread=True)

    blocking = [await measure(target, use_thread=False) for _ in range(3)]
    threaded = [await measure(target, use_thread=True) for _ in range(3)]

    def avg(rows: list[tuple[float, float]]) -> tuple[float, float]:
        return (
            sum(r[0] for r in rows) / len(rows),
            sum(r[1] for r in rows) / len(rows),
        )

    b_light, b_heavy = avg(blocking)
    t_light, t_heavy = avg(threaded)

    print("\n                   轻任务调度延迟      重任务耗时")
    print(f"  ❌ 直接在事件循环   {b_light:9.1f} ms   {b_heavy:9.1f} ms")
    print(f"  ✅ asyncio.to_thread{t_light:9.1f} ms   {t_heavy:9.1f} ms")
    print()
    if b_heavy <= 0:
        print("样本太轻，测不出来")
        return 1

    ratio = b_light / max(t_light, 0.001)
    print(f"  轻任务被拖慢: {ratio:.0f} 倍")
    print(f"  重任务自身:   {t_heavy:.1f} vs {b_heavy:.1f} ms "
          f"（差 {abs(t_heavy - b_heavy) / b_heavy * 100:.0f}% —— 本来就该几乎一样）")
    print()
    print("结论：to_thread 让【别的请求】不用等，但它没有让这个函数本身变快。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
