"""文档上传与入库接口。

POST /api/v1/documents/upload
"""

import asyncio
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.v1.schemas import UploadResponse
from app.core.ingest import index_document
from app.core.logger import logger
from app.core.parser import parse_document

router = APIRouter()

_ALLOWED_SUFFIXES = {".pdf", ".docx", ".md"}
_MAX_BYTES = 20 * 1024 * 1024


@router.post("/upload", response_model=UploadResponse, summary="上传合同并入库")
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """接收一个文件，解析 → 切分 → 编码 → 写入向量库。

    处理方式（同步、重复上传不特殊处理）的取舍写在下面 index_document 的注释里。
    """
    started = time.perf_counter()

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型 {suffix or '(无扩展名)'}，仅支持 {sorted(_ALLOWED_SUFFIXES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"文件超过 {_MAX_BYTES // 1024 // 1024} MB 上限")

    # parse_document 接收的是路径，所以先落一个临时文件；无论成败都要删掉
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)

        doc = await asyncio.to_thread(parse_document, str(tmp_path))
        # 临时文件名对用户没有意义，换回上传时的名字
        doc.filename = file.filename or doc.filename

        # ── 重复上传：Day 6 选择【什么都不做】，理由如下 ──
        #
        # 事实链：
        #   doc_id   = 文件内容的 sha1      -> 内容相同则 doc_id 相同
        #   chunk_id 由 doc_id 确定性派生   （见 D-001）
        #   写入走 upsert                    （见 D-020，内部是"墓碑 + 插入"）
        #   => 第二次上传命中同一批主键，覆盖同名行，集合行数不变
        # 实测：连传两次，doc_id 相同、written 都是 10、可查询行数始终 10。
        #
        # 所以【幂等性已经由 upsert 保证】，这里不做任何额外处理也是正确的。
        # 代价：重复上传会白付一次全量编码（几十页 PDF 约十几秒）。
        #
        # 为什么不加"先查 doc_id，存在就跳过"：
        #   · 解析器 / 切分器一改（Day 3 就修过 P1 缺陷），历史文档全部变成旧版本；
        #     跳过策略会让它们【永远无法重灌】，除非再配套 ?force=true 开关。
        #   · 收益（省几秒）要等用户反复上传才兑现，而那时这个接口早该改异步了。
        #   这是"用简单换浪费"。在 Day 6 的范围内，这是更划算的那一边。
        #
        # 同步 vs 异步（Day 6 定：先用同步）：
        #   · 实测端到端 10~15s，在 Nginx / ALB 的 60s 默认超时内，余量尚可
        #   · 真正的代价不是超时，而是这段时间占着 CPU：
        #     解析 + 编码是 CPU 密集的，GIL 决定它无法与检索请求并行，
        #     所以并发上传会让别人的提问变慢
        #   · 升级路径（W3 做，现在不实现）：接口返回 202 + task_id，
        #     任务状态存 Redis，客户端轮询 GET /documents/{task_id}。
        #     选这条路是因为 W3 本来就要引入 Redis。
        result = await index_document(doc)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(f"上传完成 {doc.filename} | {duration_ms:.0f} ms")

    return UploadResponse(
        doc_id=result["doc_id"],
        filename=result.get("filename", doc.filename),
        source_type=doc.source_type,
        chunks=result["chunks"],
        numbered=result.get("numbered", 0),
        written=result["written"],
        duration_ms=round(duration_ms, 1),
    )
