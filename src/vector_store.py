import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

from src.config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """向量存储管理器 - 单例 + 热加载 + 增量添加"""

    _instance: Optional["VectorStoreManager"] = None
    _vectorstore: Optional[Chroma] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.persist_directory = settings.VECTORSTORE_PATH
        self._load_existing()

    def _load_existing(self):
        if os.path.exists(self.persist_directory) and any(os.listdir(self.persist_directory)):
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"✅ 已加载现有向量库: {self.persist_directory}")
        else:
            self._vectorstore = None
            logger.info("向量库为空，将在首次添加时创建")

    def add_documents(self, documents: List[Document]):
        if not documents:
            return

        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self._vectorstore.add_documents(documents)

        # 新版 Chroma 已自动持久化，无需 persist()
        logger.info(f"✅ 已添加 {len(documents)} 个文档块到向量库")

    def get_retriever(self):
        if self._vectorstore is None:
            self._load_existing()
        return self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.TOP_K, "fetch_k": settings.TOP_K * 3}
        )

    def get_vectorstore(self):
        if self._vectorstore is None:
            self._load_existing()
        return self._vectorstore