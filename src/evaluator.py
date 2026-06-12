"""
自动化评估引擎。

解决原版 evaluate.py 的两个核心问题：
1. 需要人工 y/n 判断 → 改为全自动评分
2. 没有量化指标     → 输出 5 项数值指标

评分维度
─────────
semantic_score  语义相似度（0~1）
  用 Embedding 余弦相似度衡量回答与标准答案的语义接近程度。
  速度快，无额外 API 调用，阈值 ≥ 0.75 视为"正确"。

llm_judge_score  LLM 裁判分（0~3）
  用 LLM 对回答打分（0=错误 1=部分正确 2=大体正确 3=完全正确）。
  比 y/n 有更多梯度，覆盖边界情况。需要 LLM 调用，可通过 --no-llm-judge 关闭。

faithfulness  忠实度（0~1）
  LLM 判断回答中每条声明是否有来源文档支撑，量化幻觉程度。
  0 = 全部幻觉，1 = 完全有据可查。

answer_relevance  答案相关度（0~1）
  回答 Embedding 与问题 Embedding 的余弦相似度。
  衡量"有没有答非所问"。

latency_ms  端到端耗时（毫秒）

汇总指标
─────────
accuracy          语义相似度 ≥ threshold 的比例（默认 0.75）
avg_semantic      平均语义相似度
avg_llm_judge     平均 LLM 裁判分（/3）
avg_faithfulness  平均忠实度
p50/p90/p99_ms    耗时百分位
category_scores   各类别的平均语义得分
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# 语义相似度达到该阈值视为"正确"
DEFAULT_CORRECT_THRESHOLD = 0.75


class AutoEvaluator:
    """
    全自动评分器。

    依赖注入：embeddings（必须）、llm（可选）。
    不直接依赖 RAG 实例，可独立运行。
    """

    def __init__(self, embeddings, llm=None, correct_threshold: float = DEFAULT_CORRECT_THRESHOLD):
        """
        Args:
            embeddings: LangChain Embeddings 对象，用于语义相似度计算
            llm:        LangChain LLM 对象，用于 LLM 裁判和忠实度评分；为 None 时跳过
            correct_threshold: 语义相似度超过此值视为"正确"
        """
        self._embeddings = embeddings
        self._llm = llm
        self._threshold = correct_threshold

    # ══════════════════════════════════════════════════
    # 单项评分
    # ══════════════════════════════════════════════════

    def semantic_score(self, answer: str, ground_truth: str) -> float:
        """
        语义相似度：Embedding 余弦相似度。
        答案或标准答案为空时返回 0。
        """
        if not answer.strip() or not ground_truth.strip():
            return 0.0
        try:
            vecs = self._embeddings.embed_documents([answer, ground_truth])
            a, b = np.array(vecs[0]), np.array(vecs[1])
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            score = float(np.dot(a, b) / (norm_a * norm_b))
            # 余弦相似度映射到 [0, 1]
            return round(max(0.0, min(1.0, (score + 1) / 2)), 4)
        except Exception as e:
            logger.warning(f"语义相似度计算失败: {e}")
            return 0.0

    def answer_relevance(self, question: str, answer: str) -> float:
        """答案与问题的语义相似度，衡量是否答非所问"""
        return self.semantic_score(answer, question)

    def llm_judge(self, question: str, answer: str, ground_truth: str) -> dict:
        """
        LLM 裁判：用 LLM 对回答打 0-3 分。

        Returns:
            {"score": int, "reason": str}
            score: 0=错误 1=部分正确 2=大体正确 3=完全正确
        """
        if self._llm is None:
            return {"score": -1, "reason": "LLM 裁判未启用"}

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = ChatPromptTemplate.from_template(
            """你是一位严格的问答质量评审官。请根据以下信息，为 AI 回答打分。

[用户问题]
{question}

[标准答案]
{ground_truth}

[AI 回答]
{answer}

评分标准：
0 = 完全错误或与问题无关
1 = 部分正确，但遗漏了关键信息
2 = 大体正确，有轻微偏差或不够完整
3 = 完全正确，信息准确且完整

请只输出以下格式（不要有其他内容）：
SCORE: <0/1/2/3>
REASON: <一句话说明原因>"""
        )

        try:
            chain = prompt | self._llm | StrOutputParser()
            output = chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
            })

            score = -1
            reason = ""
            for line in output.strip().split("\n"):
                if line.startswith("SCORE:"):
                    try:
                        score = int(line.split(":")[1].strip())
                        score = max(0, min(3, score))
                    except ValueError:
                        pass
                elif line.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            return {"score": score, "reason": reason}
        except Exception as e:
            logger.warning(f"LLM 裁判失败: {e}")
            return {"score": -1, "reason": f"评分失败: {e}"}

    def faithfulness(self, answer: str, sources: list) -> float:
        """
        忠实度：LLM 判断回答中有多少比例的声明有来源支撑。

        Returns:
            0.0 ~ 1.0，无来源时返回 None（跳过评分）
        """
        if self._llm is None or not sources:
            return None

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        context = "\n\n".join(
            s.get("content_excerpt", "") for s in sources if s.get("content_excerpt")
        )
        if not context.strip():
            return None

        prompt = ChatPromptTemplate.from_template(
            """判断以下[AI 回答]中的每一条声明是否能在[参考来源]中找到依据。

[参考来源]
{context}

[AI 回答]
{answer}

请只输出一个 0.0~1.0 之间的小数，表示有据可查的声明占比（1.0=全部有据，0.0=全部幻觉）。
不要输出其他任何内容。"""
        )

        try:
            chain = prompt | self._llm | StrOutputParser()
            output = chain.invoke({"context": context, "answer": answer})
            score = float(output.strip())
            return round(max(0.0, min(1.0, score)), 4)
        except Exception as e:
            logger.warning(f"忠实度评分失败: {e}")
            return None

    # ══════════════════════════════════════════════════
    # 单条用例完整评分
    # ══════════════════════════════════════════════════

    def score_one(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        sources: list,
        latency_ms: float,
        category: str = "",
        enable_llm_judge: bool = True,
        enable_faithfulness: bool = True,
    ) -> dict:
        """对单条问答打完整的分，返回所有指标"""
        sem = self.semantic_score(answer, ground_truth)
        rel = self.answer_relevance(question, answer)
        judge = (
            self.llm_judge(question, answer, ground_truth)
            if enable_llm_judge
            else {"score": -1, "reason": "已跳过"}
        )
        faith = (
            self.faithfulness(answer, sources)
            if enable_faithfulness
            else None
        )

        return {
            "question":        question,
            "ground_truth":    ground_truth,
            "answer":          answer,
            "category":        category,
            "latency_ms":      round(latency_ms, 1),
            "semantic_score":  sem,
            "answer_relevance": rel,
            "llm_judge_score": judge["score"],
            "llm_judge_reason": judge["reason"],
            "faithfulness":    faith,
            "is_correct":      sem >= self._threshold,
            "sources_count":   len(sources),
        }

    # ══════════════════════════════════════════════════
    # 汇总统计
    # ══════════════════════════════════════════════════

    def summarize(self, results: list) -> dict:
        """从单条结果列表计算汇总指标"""
        if not results:
            return {}

        n = len(results)
        latencies = [r["latency_ms"] for r in results]
        sem_scores = [r["semantic_score"] for r in results]
        rel_scores = [r["answer_relevance"] for r in results]
        judge_scores = [r["llm_judge_score"] for r in results if r["llm_judge_score"] >= 0]
        faith_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]

        # 各分类平均语义分
        category_scores: dict = {}
        for r in results:
            cat = r.get("category", "未分类")
            category_scores.setdefault(cat, []).append(r["semantic_score"])
        category_avg = {
            cat: round(float(np.mean(scores)), 4)
            for cat, scores in category_scores.items()
        }

        sorted_latencies = sorted(latencies)

        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return round(data[min(idx, len(data)-1)], 1)

        return {
            "total":             n,
            "correct":           sum(1 for r in results if r["is_correct"]),
            "accuracy":          round(sum(1 for r in results if r["is_correct"]) / n, 4),
            "avg_semantic":      round(float(np.mean(sem_scores)), 4),
            "avg_relevance":     round(float(np.mean(rel_scores)), 4),
            "avg_llm_judge":     round(float(np.mean(judge_scores)) / 3, 4) if judge_scores else None,
            "avg_faithfulness":  round(float(np.mean(faith_scores)), 4) if faith_scores else None,
            "avg_latency_ms":    round(float(np.mean(latencies)), 1),
            "p50_latency_ms":    percentile(sorted_latencies, 50),
            "p90_latency_ms":    percentile(sorted_latencies, 90),
            "p99_latency_ms":    percentile(sorted_latencies, 99),
            "category_scores":   category_avg,
            "correct_threshold": self._threshold,
        }
