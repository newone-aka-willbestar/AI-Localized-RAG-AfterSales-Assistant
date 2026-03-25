# src/rag.py
# 功能：RAG 核心问答链（最简化版，适合学习）
from langchain_classic.chains import create_retrieval_chain                
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain 
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import VectorStore
from src.config import settings
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAG:
    """RAG 问答引擎（最简化版，适合学习）"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()
        
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE
        )

        # 华科制造专属 Prompt（你可以自己修改）
        system_prompt = """你是华科制造的智能客服助手。
必须基于【参考内容】回答问题。
如果不知道，就说“抱歉，此问题暂无相关信息”。
回答要专业、友好、简洁，使用中文。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("ai", "相关参考内容：{context}")
        ])

        # 官方推荐的 RAG 链
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, question: str) -> Dict[str, Any]:
        """用户提问入口"""
        logger.info(f"收到问题: {question}")
        result = self.rag_chain.invoke({"input": question})
        
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "未知") for doc in result.get("context", [])]
        }