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
        """完整流程：读取文件 → 分割文本（支持 PDF + 多种编码的 TXT）"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                # 优先 utf-8，失败再尝试 gbk（国内常见编码）
                loader = TextLoader(file_path, encoding="utf-8")
            
            docs = loader.load()
            split_docs = self.splitter.split_documents(docs)
            
            print(f"✅ 加载并分割完成：{len(split_docs)} 个文本块")
            return split_docs
            
        except UnicodeDecodeError:
            # 如果 utf-8 失败，自动尝试 gbk
            print("⚠️ utf-8 解码失败，尝试 gbk 编码...")
            loader = TextLoader(file_path, encoding="gbk")
            docs = loader.load()
            split_docs = self.splitter.split_documents(docs)
            print(f"✅ 使用 gbk 编码加载完成：{len(split_docs)} 个文本块")
            return split_docs