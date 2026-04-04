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
    """RAG 问答引擎（制造业售后服务多文档通用版）"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()
        
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE
        )

        # ================== 【修改后的通用Prompt】 ==================
        system_prompt = """你是华科制造的工业产品售后服务智能专家。

你必须严格基于【参考文档内容】回答用户问题。
- 优先使用文档中的原文或高度提炼的内容进行回答。
- 如果参考文档中没有找到相关信息，请直接回答：“根据当前上传的文档，没有找到相关内容。”
- 回答要专业、准确、简洁，使用制造业售后服务的规范术语。
- 不要使用“根据检索到的文档”这类话，直接给出答案。

参考文档内容：
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 官方推荐的 RAG 链（保持不变）
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