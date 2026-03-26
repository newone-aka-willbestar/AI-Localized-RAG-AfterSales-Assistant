# src/vector_store.py
import os
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

# 使用本地已下载的模型（不再需要联网）
EMBEDDING_MODEL_PATH = "./models/bge-small-zh-v1.5"

class VectorStore:
    def __init__(self):
        print(f"🚀 使用本地模型: {EMBEDDING_MODEL_PATH}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        self.persist_directory = "./chroma_db"
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
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})