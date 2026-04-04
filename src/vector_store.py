import os
import shutil
import logging
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import settings

logger = logging.getLogger(__name__)

# 面试话术：选择 BGE 是因为它是目前中文 RAG 领域效果最好的开源 Embedding 模型之一
EMBEDDING_MODEL_PATH = "./models/bge-small-zh-v1.5"

class VectorStore:
    def __init__(self):
        # 1. 检查模型路径，防止程序崩溃
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            logger.error(f"❌ 未找到 Embedding 模型: {EMBEDDING_MODEL_PATH}，请确保模型已下载。")
            # 这里的异常处理能体现你的工程思维
            raise FileNotFoundError(f"Model directory {EMBEDDING_MODEL_PATH} not found.")

        logger.info(f"🧬 正在加载本地 Embedding 模型: {EMBEDDING_MODEL_PATH}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            # 设置 CPU 运行，适合中小厂本地化部署环境
            model_kwargs={'device': 'cpu'} 
        )
        
        self.persist_directory = "./chroma_db"
        self.vectorstore = self._load_vectorstore()

    def _load_vectorstore(self) -> Optional[Chroma]:
        """尝试从本地加载已有的向量库"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info("📦 发现本地缓存，正在加载已有的向量库...")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None

    def add_documents(self, documents: List[Document]):
        """增量添加文档"""
        if self.vectorstore is None:
            # 第一次创建
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # 后续增量添加
            self.vectorstore.add_documents(documents)
        
        logger.info(f"✅ 向量库已更新，当前包含 {len(documents)} 条新数据")

    def clear_db(self):
        """清空向量库（方便调试和重新演示）"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            self.vectorstore = None
            logger.info("🧹 已清空本地向量库缓存")

    def get_retriever(self, k: int = 5):
        """
        获取检索器：采用 MMR 策略
        面试点：MMR 可以保证检索到的内容既相关又具有多样性，避免给 LLM 重复的信息
        """
        if self.vectorstore is None:
            # 如果库还没创建，先尝试加载一次
            self.vectorstore = self._load_vectorstore()
            if self.vectorstore is None:
                logger.warning("⚠️ 向量库为空，检索器可能无法工作。")
                return None
        
        # 面试话术：search_type="mmr" 相比普通 similarity，能减少冗余信息，节省 Token
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,                 # 最终返回给大模型的块数
                "fetch_k": 20,          # 先搜出 20 个相关的
                "lambda_mult": 0.5,     # 0.5 是平衡相关性与多样性的黄金值
            }
        )