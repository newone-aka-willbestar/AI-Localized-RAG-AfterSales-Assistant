"""
意图分类器测试。

测试策略：
- 纯规则层（embeddings=None）：确保关键词覆盖正确
- Embedding 层：用 Mock 避免加载真实模型，只测分类逻辑
- 边界：空字符串、混合关键词、大小写
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.intent_classifier import IntentClassifier, IntentResult


# ==========================================
# 辅助工具
# ==========================================

def make_classifier(with_embeddings: bool = False) -> IntentClassifier:
    """
    创建分类器实例。
    with_embeddings=False → 纯规则模式（不需要真实模型）
    with_embeddings=True  → 注入 Mock embeddings
    """
    if not with_embeddings:
        return IntentClassifier(embeddings=None)

    mock_embeddings = MagicMock()
    # embed_documents 返回固定向量（每个锚点句一条）
    mock_embeddings.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.7, 0.3, 0.0],
        [0.6, 0.4, 0.0],
    ]
    mock_embeddings.embed_query.return_value = [1.0, 0.0, 0.0]
    return IntentClassifier(embeddings=mock_embeddings)


# ==========================================
# 规则层 - 闲聊识别
# ==========================================

class TestChitchatRule:
    clf = make_classifier()

    @pytest.mark.parametrize("text", [
        "你好",
        "您好",
        "hi",
        "hello",
        "嗨",
        "哈喽",
        "谢谢",
        "谢了",
        "感谢",
        "再见",
        "拜拜",
        "bye",
    ])
    def test_greeting_farewell_classified_as_chitchat(self, text):
        result = self.clf.classify(text)
        assert result.intent == "chitchat"
        assert result.confidence == 1.0
        assert result.method == "rule"

    @pytest.mark.parametrize("text", [
        "你是谁",
        "你叫什么",
        "介绍一下你自己",
        "你是什么",
    ])
    def test_self_intro_classified_as_chitchat(self, text):
        result = self.clf.classify(text)
        assert result.intent == "chitchat"
        assert result.method == "rule"


# ==========================================
# 规则层 - 投诉识别
# ==========================================

class TestComplaintRule:
    clf = make_classifier()

    @pytest.mark.parametrize("text", [
        "我要投诉你们",
        "产品质量太差了",
        "垃圾产品，要退款",
        "非常不满意，坑爹",
        "售后差，要赔偿",
    ])
    def test_complaint_keywords_classified_correctly(self, text):
        result = self.clf.classify(text)
        assert result.intent == "complaint"
        assert result.confidence == 1.0
        assert result.method == "rule"

    def test_complaint_takes_priority_over_chitchat(self):
        """投诉关键词优先级高于闲聊"""
        # 包含"谢谢"和"不满意" → 应为 complaint
        result = self.clf.classify("谢谢但我非常不满意")
        assert result.intent == "complaint"


# ==========================================
# 规则层 - 操作指引识别
# ==========================================

class TestOperationRule:
    clf = make_classifier()

    @pytest.mark.parametrize("text", [
        "怎么安装这个设备",
        "如何启动机器",
        "操作流程是什么",
        "教我使用控制面板",
        "维修步骤有哪些",
        "怎么更换零件",
        "如何校准传感器",
    ])
    def test_operation_patterns_classified_correctly(self, text):
        result = self.clf.classify(text)
        assert result.intent == "operation"
        assert result.method == "rule"


# ==========================================
# 规则层 - 知识查询（无规则命中时的 Embedding 兜底）
# ==========================================

class TestKnowledgeQueryDefault:

    def test_no_embeddings_defaults_to_knowledge_query(self):
        """无 Embedding 时，规则未命中的问题默认为 knowledge_query"""
        clf = make_classifier(with_embeddings=False)
        result = clf.classify("产品保修期是多久")
        assert result.intent == "knowledge_query"
        assert result.method == "default"
        assert result.confidence == 0.5

    def test_error_code_query(self):
        """故障码查询 → 规则不命中 → default knowledge_query"""
        clf = make_classifier(with_embeddings=False)
        result = clf.classify("E05错误码是什么意思")
        assert result.intent == "knowledge_query"


# ==========================================
# Embedding 层
# ==========================================

class TestEmbeddingLayer:

    def test_embedding_classify_called_when_rules_miss(self):
        """规则未命中时，有 embeddings 则走 Embedding 分类"""
        mock_embeddings = MagicMock()
        # 5 个意图，每个意图 5 个锚点句 = 共 20 次 embed_documents 调用（4组各5句）
        mock_embeddings.embed_documents.return_value = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        mock_embeddings.embed_query.return_value = [1.0, 0.0, 0.0]

        clf = IntentClassifier(embeddings=mock_embeddings)
        result = clf.classify("保修条款说明")  # 规则不命中

        assert result.method == "embedding"
        assert 0.0 <= result.confidence <= 1.0

    def test_embedding_classify_picks_highest_cosine_similarity(self):
        """Embedding 层应选余弦相似度最高的意图"""
        mock_embeddings = MagicMock()
        # 为每个意图的锚点句返回固定向量
        # 让 knowledge_query 的锚点和 query 最相似
        side_effects = []
        for intent in ["knowledge_query", "operation", "complaint", "chitchat"]:
            if intent == "knowledge_query":
                # 与 query_vec 完全平行 → 余弦相似度 = 1.0
                side_effects.append([[1.0, 0.0, 0.0]] * 5)
            else:
                # 正交 → 余弦相似度 = 0.0
                side_effects.append([[0.0, 1.0, 0.0]] * 5)

        mock_embeddings.embed_documents.side_effect = side_effects
        mock_embeddings.embed_query.return_value = [1.0, 0.0, 0.0]

        clf = IntentClassifier(embeddings=mock_embeddings)
        result = clf.classify("保修条款说明")

        assert result.intent == "knowledge_query"
        assert result.method == "embedding"

    def test_embedding_fallback_on_exception(self):
        """embed_query 抛异常时，降级为 knowledge_query + fallback 方法"""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[1.0, 0.0, 0.0]] * 5
        mock_embeddings.embed_query.side_effect = RuntimeError("模型异常")

        clf = IntentClassifier(embeddings=mock_embeddings)
        result = clf.classify("查询保修期")

        assert result.intent == "knowledge_query"
        assert result.method == "fallback"

    def test_precompute_failure_falls_back_to_default(self):
        """锚点预计算失败时，分类器退化为规则+default 模式"""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.side_effect = RuntimeError("GPU OOM")

        clf = IntentClassifier(embeddings=mock_embeddings)
        # 预计算失败 → _anchor_vecs 为空 → 规则未命中时走 default
        result = clf.classify("保修期多久")
        assert result.method == "default"


# ==========================================
# IntentResult 数据结构
# ==========================================

class TestIntentResult:

    def test_intent_result_fields(self):
        r = IntentResult(intent="complaint", confidence=0.95, method="rule")
        assert r.intent == "complaint"
        assert r.confidence == 0.95
        assert r.method == "rule"

    def test_confidence_range_in_embedding(self):
        """Embedding 层输出的 confidence 必须在 [0, 1]"""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[1.0, 0.0]] * 5
        # 让 embed_query 返回与锚点相反的向量（余弦相似度=-1）
        mock_embeddings.embed_query.return_value = [-1.0, 0.0]

        clf = IntentClassifier(embeddings=mock_embeddings)
        result = clf.classify("保修期多久")

        assert 0.0 <= result.confidence <= 1.0


# ==========================================
# 边界情况
# ==========================================

class TestEdgeCases:
    clf = make_classifier()

    def test_empty_string(self):
        """空字符串不崩溃，返回合理意图"""
        result = self.clf.classify("")
        assert result.intent in ("knowledge_query", "chitchat", "operation", "complaint")

    def test_whitespace_only(self):
        """全空格不崩溃"""
        result = self.clf.classify("   ")
        assert result.intent is not None

    def test_very_long_input(self):
        """超长输入不崩溃"""
        long_text = "设备故障 " * 500
        result = self.clf.classify(long_text)
        assert result.intent is not None

    def test_mixed_language(self):
        """中英文混合"""
        result = self.clf.classify("The 设备 error code E05 是什么")
        assert result.intent is not None

    def test_punctuation_only(self):
        """纯标点"""
        result = self.clf.classify("！！！？？？")
        assert result.intent is not None
