"""
HyDE (Hypothetical Document Embeddings) 检索增强模块。

原理：
  用户提问（短句）与知识库文档（长段落）的 embedding 分布差异很大，
  直接用问题做向量检索召回率偏低。
  HyDE 先让 LLM 生成一段"风格接近真实手册"的假设文档，
  再用这段文本做向量检索——embedding 空间更接近，召回率更高。

示例：
  输入:  "迁移学习的优势是什么？"
  输出:  "迁移学习指将在大规模数据集上预训练的模型参数迁移到目标任务。
          其优势包括：减少对标注数据的依赖、加快收敛速度、
          在小样本场景下仍能获得较高准确率..."
  效果:  输出的 embedding 与文献中的论述段落高度相似

代价：每次问答多一次 LLM 调用（约 1-3 秒），可通过 HYDE_ENABLED=false 关闭。
"""
import logging

logger = logging.getLogger(__name__)

# 通用知识助手 HyDE Prompt
# 要求生成结果贴近知识库文档风格，提升向量检索召回率
_PROMPT_TEMPLATE = """\
你是一个专业的 AI 知识助手。
请针对用户的问题，写一段简短的、百科或文献风格的标准回答段落。

要求：
1. 内容客观、准确，覆盖问题的核心要点
2. 语言专业，结构清晰，避免口语化表达
3. 不要有"你好"、"建议您"等客套话
4. 长度控制在 100-200 字之间

用户问题：{question}

请输出知识段落："""


class HyDE:
    """
    假设文档生成器。

    设计要点：
    - 接受 LLM 对象注入（由 RAG 传入），避免与 rag.py 循环导入
    - generate() 内置降级保护：LLM 调用失败时返回原始问题，不中断主流程
    - 所有 langchain 依赖懒加载，避免模块顶层 import 的 pydantic_v1 兼容问题
    """

    def __init__(self, llm):
        """
        Args:
            llm: 已经初始化的 LangChain LLM 对象（由 RAG.__init__ 传入）
        """
        self.llm = llm

    def generate(self, question: str) -> str:
        """
        将用户问题变换为假设文档。

        chain.invoke 传入 run_name="HyDE.generate"，LangSmith 追踪树中
        会独立显示此节点，可以看到输入问题、输出假设文档及 token 消耗。

        Returns:
            str: 假设文档文本（成功时）或原始问题（失败时降级）
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        try:
            prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
            chain = prompt | self.llm | StrOutputParser()
            hypothetical_doc = chain.invoke(
                {"question": question},
                config={"run_name": "HyDE.generate"},
            )
            result = hypothetical_doc.strip()
            logger.info(
                f"HyDE 变换完成，原问题长度={len(question)}，"
                f"假设文档长度={len(result)}"
            )
            return result

        except Exception as e:
            logger.warning(f"HyDE 生成失败（将使用原问题检索）: {e}")
            return question
