from fastapi import FastAPI, File, UploadFile
from loguru import logger
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="ContractLens API",
    description="基于 GraphRAG 的智能合同审查引擎",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {"message": "ContractLens is running!", "status": "healthy"}


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    合同上传接口
    暂时只返回文件基本信息，验证 FastAPI 的异步文件处理能力
    """
    logger.info(f"收到文件：{file.filename}")
    return JSONResponse(
        status_code=200,
        content={
            "filename": file.filename,
            "content_type": file.content_type,
            "message": "文件接收成功",
        },
    )
@app.post("/api/v1/chat")
async def chat(query:str):
    """
    合同问答接口
    """
    return {"question":query,"answer":"（Day 6 将替换为真正的 RAG 回答）","sources":[]}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
