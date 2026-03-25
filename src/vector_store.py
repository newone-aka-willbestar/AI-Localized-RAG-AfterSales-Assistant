import os
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

from src.config import settings

logger = logging.getLogger(__name__)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 使用国内镜像
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
        self.persist_directory = settings.VECTORSTORE_PATH
        self.vectorstore = None

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
        logger.info(f"✅ 已添加 {len(documents)} 个文档块")

    def get_retriever(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.TOP_K}
        )