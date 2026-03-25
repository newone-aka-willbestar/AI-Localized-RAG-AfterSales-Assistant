# src/vector_store.py
# 功能：向量数据库管理（RAG 第二步：向量化 + 存储）
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from src.config import settings
import logging
import os

logger = logging.getLogger(__name__)

# 国内加速（解决 huggingface 下载卡住）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class VectorStore:
    """向量存储 - 负责把文本块变成向量并保存"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.persist_directory = settings.VECTORSTORE_PATH
        self.vectorstore = None

    def add_documents(self, documents: List[Document]):
        """添加文档到向量库（第一次创建，后续自动追加）"""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        
        logger.info(f"✅ 已添加 {len(documents)} 个文档块到向量库")

    def get_retriever(self):
        """获取检索器（RAG 检索阶段使用）"""
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.TOP_K}
        )