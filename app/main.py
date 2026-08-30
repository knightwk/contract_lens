import sys

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import PydanticValidationError, register_exception_handlers
from app.utils.pfd_parser import extract_text_from_pdf_bytes

logger.remove()
logger.add(sys.stdout, level="INFO", backtrace=False, diagnose=False)

app = FastAPI(
    title="ContractLens API",
    description="基于 GraphRAG 的智能合同审查引擎",
    version="0.1.0",
)
register_exception_handlers(app)


@app.get("/")
async def root():
    return {"message": "ContractLens is running!", "status": "healthy"}


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    合同上传接口
    暂时只返回文件基本信息，验证 FastAPI 的异步文件处理能力
    """
    # 1. 校验文件后缀
    if not file.filename.lower().endswith(".pdf"):
        raise PydanticValidationError("仅支持 PDF 文件上传")

    # 2. 读取字节流
    contents = await file.read()

    # 3. 调用解析函数
    extract_text, total_pages = extract_text_from_pdf_bytes(file_bytes=contents)

    # 4. 生成预览
    length = len(extract_text)
    preview = extract_text[: min(length, 100)] + ("..." if length > 100 else "")

    # 5. 返回响应
    logger.info(f"收到文件：{file.filename}")
    return JSONResponse(
        status_code=200,
        content={
            "filename": file.filename,
            "total_pages": total_pages,
            "text_length": length,
            "preview": preview,
            "content_type": file.content_type,
            "message": "PDF 解析成功",
            "is_scanned": length == 0,
        },
    )


@app.post("/api/v1/chat")
async def chat(query: str):
    """
    合同问答接口
    """
    return {
        "question": query,
        "answer": "（Day 6 将替换为真正的 RAG 回答）",
        "sources": [],
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
