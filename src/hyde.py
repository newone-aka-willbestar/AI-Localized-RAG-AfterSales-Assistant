import logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings

logger = logging.getLogger(__name__)

class HyDE:
    """
    HyDE (Hypothetical Document Embeddings) 
    作用：将用户简单的提问“翻译”成一段模拟的技术手册段落，提高检索召回率。
    """
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.3  # 稍微给一点创造力，但不要太高
        )
        
        # 优化点：针对你的“工业售后”场景定制 Prompt
        # 面试时说：我专门调整了 Prompt，让 AI 模仿技术手册的口吻生成假答案
        self.prompt = ChatPromptTemplate.from_template("""
        你是一个专业的工业设备维修专家。请针对以下用户咨询的问题，写一段简短的、类似于技术维修手册中的标准回答。
        要求：
        1. 包含可能的故障原因。
        2. 包含标准的检查或修复步骤。
        3. 语言专业、客观，不要有“你好”、“建议”等客套话。
        
        用户问题：{question}
        
        模拟手册段落示例：
        [故障现象] 设备运行中出现异响。
        [可能原因] 轴承磨损、异物进入或润滑不足。
        [检查步骤] 1. 停机断电；2. 检查轴承间隙；3. 清理腔体并添加润滑油。
        
        请开始生成：
        """)

    def generate(self, question: str) -> str:
        """
        生成假设文档。如果生成失败，降级返回原问题。
        """
        try:
            logger.info(f"🚀 正在为问题执行 HyDE 变换: {question}")
            
            chain = self.prompt | self.llm | StrOutputParser()
            hypothetical_doc = chain.invoke({"question": question})
            
            # 去除可能的多余空格
            return hypothetical_doc.strip()
            
        except Exception as e:
            logger.error(f"❌ HyDE 生成失败: {e}，将使用原提问进行检索。")
            return question