from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag import RAG
from src.config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="华科制造 AI 智能客服")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAG()   # 全局 RAG 实例

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档 → 向量化"""
    suffix = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = DocumentLoader()
        docs = loader.load_and_split(tmp_path)
        
        vector_store = VectorStore()
        vector_store.add_documents(docs)
        
        logger.info(f"✅ 文档 {file.filename} 上传成功，处理 {len(docs)} 个块")
        return {"message": f"✅ 上传成功！已处理 {len(docs)} 个文本块", "filename": file.filename}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/ask")
async def ask(request: QuestionRequest):
    """提问（已增强：返回context用于调试）"""
    try:
        # 调用 RAG（假设你的 RAG.ask 支持返回 context，如果不支持请告诉我）
        result = rag.ask(request.question)
        
        # 如果你的 RAG.ask 只返回字符串，我们这里加一层包装方便调试
        if isinstance(result, str):
            return {
                "answer": result,
                "note": "（调试模式：请把 src/rag.py 的 prompt 换成上面我给的版本）"
            }
        else:
            # 如果 RAG 已经返回 dict，就直接返回
            return result
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "简化版 RAG 系统已启动"}