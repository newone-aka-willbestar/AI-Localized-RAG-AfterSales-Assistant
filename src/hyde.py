from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.config import settings

class HyDE:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.2
        )

    def generate_hypothetical_document(self, question: str) -> str:
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""请针对以下问题，写一段假设的、像知识库文档一样的答案（只输出内容，不要加说明）：
问题：{question}
假设答案："""
        )
        chain = prompt | self.llm
        response = chain.invoke({"question": question})
        return response.content