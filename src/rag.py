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
    """RAG 问答引擎"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = self.vector_store.get_retriever()
        
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,          # 降低温度，回答更确定
            num_predict=1024
        )

        
        system_prompt = """你是华科制造的工业产品售后服务智能专家。

【核心规则 - 必须严格遵守】
- 你只能使用【参考文档内容】中的信息来回答问题。
- 禁止使用任何外部知识、个人经验、通用行业做法或编造内容。
- 如果参考文档中没有明确相关信息，必须原样回答：“根据当前上传的文档，没有找到相关内容。”
- 回答要专业、准确、简洁，尽量使用文档中的原文表述。
- 不要添加“根据检索到的文档”“推荐性内容”等多余前缀，直接给出答案。

参考文档内容：
{context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

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