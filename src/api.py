from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import logging
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag import RAG
from src.config import settings

# 初始化日志
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="华科制造 AI 智能客服")

# 允许跨域（生产环境建议限制域名）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. 全局 RAG 实例（初次启动时可能没有文档）
rag = RAG()

# 模拟一个简单的 API Key 校验（面试加分点：体现安全意识）
# 如果你的 Streamlit 里传了 key，这里如果不校验会报错
async def verify_api_key(x_api_key: str = Header(None)):
    # 这里的 key 建议和 Streamlit/settings 保持一致
    if x_api_key != "your-secret-key-2026":
        raise HTTPException(status_code=403, detail="无效的 API Key")
    return x_api_key

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    """
    上传文档 → 向量化 → 更新检索引擎
    """
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="目前仅支持 PDF 格式文档")
    
    # 使用临时文件存储上传内容
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # A. 加载并切分文档
        loader = DocumentLoader()
        docs = loader.load_and_split(tmp_path)
        
        # B. 存入向量库
        vector_store = VectorStore()
        vector_store.add_documents(docs)
        
        # C. 【关键修改】通知 RAG 实例使用新文档初始化检索器
        # 这一步如果不做，RAG 里的 BM25 和精排层就无法工作
        rag.init_retriever(docs)
        
        logger.info(f"✅ 文档 {file.filename} 处理完成：已更新混合检索引擎")
        return {
            "message": f"✅ 上传成功！已处理 {len(docs)} 个文本块并更新索引", 
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"❌ 上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/ask")
async def ask(request: QuestionRequest, api_key: str = Depends(verify_api_key)):
    """
    执行 RAG 问答
    """
    try:
        # 调用已经重构过的 rag.ask，它现在返回一个包含 answer 和 sources 的字典
        result = rag.ask(request.question)
        
        # 统一返回结果格式
        return result
        
    except Exception as e:
        logger.error(f"❌ 问答链路异常: {e}")
        # 给前端一个友好的报错，面试时可以说考虑了系统的健壮性
        return {
            "answer": "抱歉，系统处理您的提问时遇到了点问题，请稍后再试。",
            "sources": []
        }

@app.get("/health")
async def health():
    """健康检查接口"""
    return {
        "status": "online", 
        "engine_ready": rag.final_retriever is not None,
        "message": "华科制造 RAG 系统运行中"
    }