# src/document_loader.py
# 功能：加载 PDF/TXT 并自动分割成小块（RAG 第一步）
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from src.config import settings

class DocumentLoader:
    """文档加载器 - 负责读取文件并分割成适合 RAG 的小块"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def load_and_split(self, file_path: str) -> List[Document]:
        """完整流程：读取文件 → 分割文本"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
        
        docs = loader.load()
        split_docs = self.splitter.split_documents(docs)
        
        print(f"✅ 加载并分割完成：{len(split_docs)} 个文本块")
        return split_docs