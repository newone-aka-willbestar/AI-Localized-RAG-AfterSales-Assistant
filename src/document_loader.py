from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os

class DocumentProcessor:
    """处理文档加载与分割"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """根据文件扩展名自动选择加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif ext == '.md':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        return self.text_splitter.split_documents(documents)
    
    def process(self, file_path: str) -> List[Document]:
        """完整流程：加载+分割"""
        docs = self.load_documents(file_path)
        return self.split_documents(docs)