"""
AutoEvaluator 测试。

策略：
- Embedding 和 LLM 全部 Mock，不做真实 API 调用
- 验证：各维度评分逻辑、汇总统计、边界情况
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.evaluator import AutoEvaluator, DEFAULT_CORRECT_THRESHOLD


# ==========================================
# 测试夹具
# ==========================================

def make_evaluator(with_llm=False):
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # 完全相同 → 相似度 1.0
    ]
    mock_emb.embed_query.return_value = [1.0, 0.0, 0.0]
    llm = MagicMock() if with_llm else None
    return AutoEvaluator(embeddings=mock_emb, llm=llm)


# ==========================================
# semantic_score
# ==========================================

class TestSemanticScore:

    def test_identical_vectors_score_one(self):
        ev = make_evaluator()
        score = ev.semantic_score("答案", "答案")
        assert score == 1.0

    def test_orthogonal_vectors_score_half(self):
        ev = make_evaluator()
        ev._embeddings.embed_documents.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],  # 余弦 = 0 → 归一化 = 0.5
        ]
        score = ev.semantic_score("A", "B")
        assert abs(score - 0.5) < 0.01

    def test_opposite_vectors_score_zero(self):
        ev = make_evaluator()
        ev._embeddings.embed_documents.return_value = [
            [1.0, 0.0],
            [-1.0, 0.0],  # 余弦 = -1 → 归一化 = 0
        ]
        score = ev.semantic_score("A", "B")
        assert score == 0.0

    def test_empty_answer_returns_zero(self):
        ev = make_evaluator()
        assert ev.semantic_score("", "标准答案") == 0.0

    def test_empty_ground_truth_returns_zero(self):
        ev = make_evaluator()
        assert ev.semantic_score("回答内容", "") == 0.0

    def test_whitespace_only_returns_zero(self):
        ev = make_evaluator()
        assert ev.semantic_score("   ", "标准答案") == 0.0

    def test_embedding_failure_returns_zero(self):
        ev = make_evaluator()
        ev._embeddings.embed_documents.side_effect = RuntimeError("模型崩溃")
        assert ev.semantic_score("答案", "标准") == 0.0

    def test_score_always_in_zero_one_range(self):
        ev = make_evaluator()
        for _ in range(5):
            score = ev.semantic_score("任意文本", "另一段文本")
            assert 0.0 <= score <= 1.0


# ==========================================
# answer_relevance
# ==========================================

class TestAnswerRelevance:

    def test_returns_float_in_range(self):
        ev = make_evaluator()
        score = ev.answer_relevance("这是问题", "这是答案")
        assert 0.0 <= score <= 1.0

    def test_delegates_to_semantic_score(self):
        """answer_relevance 本质是对调参数的 semantic_score"""
        ev = make_evaluator()
        r1 = ev.answer_relevance("问题", "答案")
        r2 = ev.semantic_score("答案", "问题")
        assert r1 == r2


# ==========================================
# llm_judge
# ==========================================

class TestLLMJudge:

    def test_no_llm_returns_minus_one(self):
        ev = make_evaluator(with_llm=False)
        result = ev.llm_judge("问题", "答案", "标准")
        assert result["score"] == -1
        assert "reason" in result

    def test_result_has_required_fields(self):
        ev = make_evaluator(with_llm=True)
        # ChatPromptTemplate 在函数内懒加载，patch 原始模块路径
        with patch("langchain_core.prompts.ChatPromptTemplate") as mock_pt:
            chain = MagicMock()
            chain.invoke.return_value = "SCORE: 3\nREASON: 完全正确"
            mock_pt.from_template.return_value.__or__ = MagicMock(return_value=chain)
            result = ev.llm_judge("问题", "答案", "标准答案")
        assert "score" in result
        assert "reason" in result

    def test_llm_exception_returns_minus_one(self):
        ev = make_evaluator(with_llm=True)
        with patch("langchain_core.prompts.ChatPromptTemplate") as mock_pt:
            chain = MagicMock()
            chain.invoke.side_effect = RuntimeError("API 超时")
            mock_pt.from_template.return_value.__or__ = MagicMock(return_value=chain)
            result = ev.llm_judge("问题", "答案", "标准")
        assert result["score"] == -1


# ==========================================
# faithfulness
# ==========================================

class TestFaithfulness:

    def test_no_llm_returns_none(self):
        ev = make_evaluator(with_llm=False)
        assert ev.faithfulness("答案", [{"content_excerpt": "来源"}]) is None

    def test_empty_sources_returns_none(self):
        ev = make_evaluator(with_llm=True)
        assert ev.faithfulness("答案", []) is None

    def test_blank_content_excerpt_returns_none(self):
        ev = make_evaluator(with_llm=True)
        assert ev.faithfulness("答案", [{"content_excerpt": ""}]) is None

    def test_sources_without_excerpt_key_returns_none(self):
        ev = make_evaluator(with_llm=True)
        assert ev.faithfulness("答案", [{"source": "doc.pdf"}]) is None


# ==========================================
# score_one
# ==========================================

class TestScoreOne:

    def test_returns_all_required_fields(self):
        ev = make_evaluator()
        result = ev.score_one(
            question="问题", answer="回答", ground_truth="标准",
            sources=[], latency_ms=123.4,
        )
        required = {
            "question", "ground_truth", "answer", "category",
            "latency_ms", "semantic_score", "answer_relevance",
            "llm_judge_score", "llm_judge_reason", "faithfulness",
            "is_correct", "sources_count",
        }
        assert required.issubset(result.keys())

    def test_is_correct_true_above_threshold(self):
        ev = make_evaluator()
        # 相同向量 → semantic_score = 1.0 ≥ 0.75
        result = ev.score_one("q", "a", "a", [], 100)
        assert result["is_correct"] is True

    def test_is_correct_false_below_threshold(self):
        ev = make_evaluator()
        ev._embeddings.embed_documents.return_value = [
            [1.0, 0.0], [0.0, 1.0],  # 正交 → 0.5 < 0.75
        ]
        result = ev.score_one("q", "A", "B", [], 100)
        assert result["is_correct"] is False

    def test_latency_recorded_correctly(self):
        ev = make_evaluator()
        result = ev.score_one("q", "a", "a", [], latency_ms=456.7)
        assert result["latency_ms"] == 456.7

    def test_sources_count_recorded(self):
        ev = make_evaluator()
        sources = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
        result = ev.score_one("q", "a", "a", sources, 100)
        assert result["sources_count"] == 3

    def test_category_recorded(self):
        ev = make_evaluator()
        result = ev.score_one("q", "a", "a", [], 100, category="方法论")
        assert result["category"] == "方法论"

    def test_llm_judge_skipped_when_disabled(self):
        ev = make_evaluator(with_llm=True)
        result = ev.score_one("q", "a", "a", [], 100, enable_llm_judge=False)
        assert result["llm_judge_score"] == -1

    def test_faithfulness_skipped_when_disabled(self):
        ev = make_evaluator(with_llm=True)
        result = ev.score_one("q", "a", "a", [], 100, enable_faithfulness=False)
        assert result["faithfulness"] is None


# ==========================================
# summarize
# ==========================================

class TestSummarize:

    def _make_results(self, n_correct, n_wrong):
        results = []
        for _ in range(n_correct):
            results.append({
                "semantic_score": 0.9, "answer_relevance": 0.8,
                "llm_judge_score": 3, "faithfulness": 0.95,
                "latency_ms": 200.0, "is_correct": True, "category": "正确类",
            })
        for _ in range(n_wrong):
            results.append({
                "semantic_score": 0.4, "answer_relevance": 0.5,
                "llm_judge_score": 1, "faithfulness": 0.3,
                "latency_ms": 400.0, "is_correct": False, "category": "错误类",
            })
        return results

    def test_empty_results_returns_empty_dict(self):
        ev = make_evaluator()
        assert ev.summarize([]) == {}

    def test_accuracy_calculation(self):
        ev = make_evaluator()
        summary = ev.summarize(self._make_results(n_correct=3, n_wrong=1))
        assert abs(summary["accuracy"] - 0.75) < 0.01

    def test_correct_and_total_count(self):
        ev = make_evaluator()
        summary = ev.summarize(self._make_results(n_correct=2, n_wrong=3))
        assert summary["correct"] == 2
        assert summary["total"] == 5

    def test_avg_semantic_score(self):
        ev = make_evaluator()
        # (0.9 + 0.4) / 2 = 0.65
        summary = ev.summarize(self._make_results(n_correct=1, n_wrong=1))
        assert abs(summary["avg_semantic"] - 0.65) < 0.01

    def test_latency_percentile_order(self):
        ev = make_evaluator()
        results = [
            {"semantic_score": 0.8, "answer_relevance": 0.7,
             "llm_judge_score": -1, "faithfulness": None,
             "latency_ms": float(ms), "is_correct": True, "category": "A"}
            for ms in range(100, 200, 10)  # 10 个数据点
        ]
        summary = ev.summarize(results)
        assert summary["p50_latency_ms"] <= summary["p90_latency_ms"]
        assert summary["p90_latency_ms"] <= summary["p99_latency_ms"]

    def test_category_scores_all_present(self):
        ev = make_evaluator()
        summary = ev.summarize(self._make_results(n_correct=2, n_wrong=2))
        assert "category_scores" in summary
        assert "正确类" in summary["category_scores"]
        assert "错误类" in summary["category_scores"]

    def test_avg_llm_judge_none_when_all_minus_one(self):
        ev = make_evaluator()
        results = [{
            "semantic_score": 0.8, "answer_relevance": 0.7,
            "llm_judge_score": -1, "faithfulness": None,
            "latency_ms": 100.0, "is_correct": True, "category": "A",
        }]
        assert ev.summarize(results)["avg_llm_judge"] is None

    def test_avg_faithfulness_none_when_all_none(self):
        ev = make_evaluator()
        results = [{
            "semantic_score": 0.8, "answer_relevance": 0.7,
            "llm_judge_score": -1, "faithfulness": None,
            "latency_ms": 100.0, "is_correct": True, "category": "A",
        }]
        assert ev.summarize(results)["avg_faithfulness"] is None

    def test_correct_threshold_recorded_in_summary(self):
        ev = make_evaluator()
        summary = ev.summarize(self._make_results(1, 0))
        assert summary["correct_threshold"] == DEFAULT_CORRECT_THRESHOLD
