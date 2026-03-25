# src/api.py
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

app = FastAPI(title="华科制造 AI 智能客服（简化学习版）")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rag = RAG()                     # 全局 RAG 实例（学习阶段够用）

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档 → 向量化"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = DocumentLoader()
    docs = loader.load_and_split(tmp_path)
    
    vector_store = VectorStore()
    vector_store.add_documents(docs)
    
    os.unlink(tmp_path)
    return {"message": f"✅ 上传成功！已处理 {len(docs)} 个文本块"}

@app.post("/ask")
async def ask(request: QuestionRequest):
    """提问"""
    try:
        result = rag.ask(request.question)
        return result
    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "简化版 RAG 系统已启动"}