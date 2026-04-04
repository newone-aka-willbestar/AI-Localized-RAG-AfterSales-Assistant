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
    suffix = os.path.splitext(file.filename)[1].lower()
    
    # 1. 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 2. 执行加载
        loader = DocumentLoader()
        docs = loader.load_and_split(tmp_path)
        
        # 3. 存入向量库
        vector_store = VectorStore()
        vector_store.add_documents(docs)
        
        # 4. 更新 RAG 检索器
        rag.init_retriever(docs)
        
        return {"message": "上传成功", "filename": file.filename}

    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 关键修复：先显式尝试清理垃圾回收，释放文件句柄
        import gc
        gc.collect() 
        
        # 增加防御性删除
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path) # 使用 os.remove 比 os.unlink 在 Windows 上更稳一点
        except Exception as delete_error:
            logger.warning(f"⚠️ 临时文件清理失败（但不影响业务）: {delete_error}")

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