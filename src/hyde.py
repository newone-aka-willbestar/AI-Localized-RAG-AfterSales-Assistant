"""
HyDE (Hypothetical Document Embeddings) 检索增强模块。

原理：
  用户提问（短句）与知识库文档（长段落）的 embedding 分布差异很大，
  直接用问题做向量检索召回率偏低。
  HyDE 先让 LLM 生成一段"风格接近真实手册"的假设文档，
  再用这段文本做向量检索——embedding 空间更接近，召回率更高。

示例：
  输入:  "设备不启动怎么办？"
  输出:  "[故障现象] 设备上电后无任何响应。
          [可能原因] 电源熔断器熔断、主控板故障或急停开关未复位。
          [检查步骤] 1. 检查熔断器..."
  效果:  输出的 embedding 与手册中的故障描述章节高度相似

代价：每次问答多一次 LLM 调用（约 1-3 秒），可通过 HYDE_ENABLED=false 关闭。
"""
import logging

logger = logging.getLogger(__name__)

# 针对工业售后场景定制的 Prompt
# 要求生成结果必须符合技术手册格式，这样 embedding 才能与手册内容对齐
_PROMPT_TEMPLATE = """\
你是一个专业的工业设备维修专家。
请针对用户的问题，写一段简短的、技术手册风格的标准回答。

要求：
1. 包含可能的故障原因
2. 包含标准的检查或修复步骤
3. 语言专业、客观，使用"[故障现象]"、"[可能原因]"、"[检查步骤]"等标准字段格式
4. 不要有"你好"、"建议"等客套话

用户问题：{question}

请输出标准手册段落："""


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
