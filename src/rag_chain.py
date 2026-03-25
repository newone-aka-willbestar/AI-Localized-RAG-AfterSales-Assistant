
from langchain.chains import create_retrieval_chain                    
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain  
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.vector_store import VectorStoreManager
from src.hyde import HyDE
from src.config import settings
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, vector_manager: VectorStoreManager = None):
        self.vector_manager = vector_manager or VectorStoreManager()
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.TEMPERATURE
        )
        self.retriever = self.vector_manager.get_retriever()
        self.hyde = HyDE() if settings.USE_HYDE else None

        system_prompt = """你是**华科制造**官方智能售后客服助手。
请严格根据参考内容，用专业、友好、规范的语气回答。
必须做到：
1. 直接引用产品型号、故障代码、保修条款。
2. 回答末尾标注来源。
3. 无法回答时礼貌引导用户提供更多信息或转人工。

【参考内容】
{context}

【用户问题】
{input}
【回答】"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def _retrieve_with_hyde(self, question: str):
        if not self.hyde or not settings.USE_HYDE:
            return self.retriever.invoke(question)
        # HyDE 增强
        hypo_doc = self.hyde.generate_hypothetical_document(question)
        hypo_results = self.retriever.invoke(hypo_doc)
        original_results = self.retriever.invoke(question)
        combined = hypo_results + original_results
        seen = set()
        unique = []
        for doc in combined:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique[:settings.TOP_K * 2]

    def answer(self, question: str) -> Dict[str, Any]:
        logger.info(f"收到问题: {question}")
        docs = self._retrieve_with_hyde(question)
        result = self.rag_chain.invoke({"input": question})
        return {
            "answer": result["answer"],
            "sources": [
                {"content": doc.page_content[:300] + "...", "metadata": doc.metadata}
                for doc in docs
            ]
        }