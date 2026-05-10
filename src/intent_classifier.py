"""
意图分类器模块。

将用户输入分为四类，决定后续走哪条处理路径：

  knowledge_query  →  走完整 RAG 流程（HyDE + 检索 + LLM）
  operation        →  走完整 RAG 流程（操作指引类，同上）
  complaint        →  走完整 RAG 流程 + 标记投诉标签（供后续分析）
  chitchat         →  跳过 RAG，直接 LLM 随意回复，不消耗检索资源

分类策略：关键词规则优先，覆盖不到的用 Embedding 余弦相似度兜底。
- 规则层：速度 <1ms，覆盖约 70% 场景（问候、明显闲聊、投诉词）
- Embedding 层：复用已加载的 bge 模型，无额外开销，覆盖剩余 30%

为什么不用 LLM 分类？
  每次多一次 DeepSeek API 调用（0.1-0.3 秒 + 费用），对简单分类得不偿失。
"""
import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Intent = Literal["knowledge_query", "operation", "complaint", "chitchat"]


@dataclass
class IntentResult:
    intent: Intent
    confidence: float   # 0.0 ~ 1.0，规则命中时为 1.0
    method: str         # "rule" 或 "embedding"


# ==========================================
# 关键词规则（按优先级排列）
# ==========================================

_CHITCHAT_PATTERNS = [
    r"^(你好|您好|hi|hello|嗨|哈喽)[!！。,，\s]*$",
    r"^(谢谢|谢了|感谢|thank)[!！。,，\s]*$",
    r"^(再见|拜拜|bye|晚安|下班了)[!！。,，\s]*$",
    r"^(你是谁|你叫什么|你是什么|介绍一下你自己)",
    r"^(天气|今天几号|几点了|心情)",
    r"^(聊聊|讲个笑话|说个故事|随便聊)",
]

_COMPLAINT_KEYWORDS = [
    "投诉", "举报", "太差", "垃圾", "坑爹", "骗人", "退款", "赔偿",
    "不满意", "失望", "坏了", "出问题", "质量差", "售后差", "态度差",
]

_OPERATION_PATTERNS = [
    r"(怎么|如何|步骤|操作|流程|方法|教我|教一下).{0,10}(使用|安装|启动|拆卸|维修|更换|校准|设置|配置)",
    r"(使用|安装|启动|拆卸|维修|更换|校准|设置|配置).{0,10}(怎么|如何|步骤|操作|流程|方法)",
    r"(第.步|步骤\d|操作手册|操作规程)",
]

# Embedding 兜底用的意图锚点句（各取代表性表达）
_INTENT_ANCHORS: dict[Intent, list[str]] = {
    "knowledge_query": [
        "产品保修期是多久",
        "错误代码E05是什么意思",
        "设备最大承载重量是多少",
        "这个型号支持哪些功能",
        "备件在哪里购买",
    ],
    "operation": [
        "如何安装这个设备",
        "怎么操作控制面板",
        "维修步骤是什么",
        "怎么更换零件",
        "启动流程是怎样的",
    ],
    "complaint": [
        "我要投诉你们的产品质量",
        "售后服务太差了",
        "这个东西坏了要退款",
        "非常不满意",
        "产品有质量问题",
    ],
    "chitchat": [
        "你好啊今天天气怎么样",
        "聊聊天吧",
        "你是谁啊",
        "随便说点什么",
        "谢谢你的帮助",
    ],
}


class IntentClassifier:
    """
    两层意图分类器：关键词规则 → Embedding 相似度。

    复用 RAG 的 Embedding 模型（由外部注入），不额外加载模型。
    embeddings 参数传 None 时退化为纯规则模式（测试友好）。
    """

    def __init__(self, embeddings=None):
        """
        Args:
            embeddings: LangChain Embeddings 对象（由 api.py 从 VectorStore 取出注入）
                        传 None 时只用规则层，不做 Embedding 兜底
        """
        self._embeddings = embeddings
        self._anchor_vecs: dict[Intent, list] = {}

        if embeddings is not None:
            self._precompute_anchors()

    def _precompute_anchors(self) -> None:
        """启动时预计算锚点句的向量，分类时直接做余弦相似度，不重复 encode"""
        try:
            import numpy as np
            for intent, sentences in _INTENT_ANCHORS.items():
                vecs = self._embeddings.embed_documents(sentences)
                self._anchor_vecs[intent] = [np.array(v) for v in vecs]
            logger.info("意图分类器锚点向量预计算完成")
        except Exception as e:
            logger.warning(f"锚点预计算失败，将退化为纯规则模式: {e}")
            self._anchor_vecs = {}

    def classify(self, text: str) -> IntentResult:
        """
        对输入文本进行意图分类。

        Returns:
            IntentResult，包含 intent、confidence、method
        """
        text = text.strip()

        # 第一层：关键词规则
        result = self._rule_classify(text)
        if result is not None:
            return result

        # 第二层：Embedding 相似度兜底
        if self._anchor_vecs:
            return self._embedding_classify(text)

        # 无 Embedding：默认 knowledge_query
        logger.debug("无 Embedding 模型，默认分类为 knowledge_query")
        return IntentResult(intent="knowledge_query", confidence=0.5, method="default")

    def _rule_classify(self, text: str) -> IntentResult | None:
        """关键词规则层，命中返回结果，未命中返回 None"""
        lower = text.lower()

        # 投诉词优先（避免被其他规则误判）
        for kw in _COMPLAINT_KEYWORDS:
            if kw in text:
                return IntentResult(intent="complaint", confidence=1.0, method="rule")

        # 闲聊模式
        for pattern in _CHITCHAT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return IntentResult(intent="chitchat", confidence=1.0, method="rule")

        # 操作指引
        for pattern in _OPERATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return IntentResult(intent="operation", confidence=1.0, method="rule")

        return None

    def _embedding_classify(self, text: str) -> IntentResult:
        """Embedding 余弦相似度分类，返回得分最高的意图"""
        import numpy as np

        try:
            query_vec = np.array(self._embeddings.embed_query(text))
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return IntentResult(intent="knowledge_query", confidence=0.5, method="embedding")

            best_intent: Intent = "knowledge_query"
            best_score = -1.0

            for intent, vecs in self._anchor_vecs.items():
                scores = []
                for anchor_vec in vecs:
                    anchor_norm = np.linalg.norm(anchor_vec)
                    if anchor_norm == 0:
                        continue
                    cos_sim = float(np.dot(query_vec, anchor_vec) / (query_norm * anchor_norm))
                    scores.append(cos_sim)
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_intent = intent

            confidence = min(1.0, max(0.0, (best_score + 1) / 2))  # [-1,1] → [0,1]
            logger.debug(f"Embedding 分类: {best_intent} (score={best_score:.3f})")
            return IntentResult(intent=best_intent, confidence=confidence, method="embedding")

        except Exception as e:
            logger.warning(f"Embedding 分类失败，回退 knowledge_query: {e}")
            return IntentResult(intent="knowledge_query", confidence=0.5, method="fallback")
