"""
全自动 RAG 评估脚本。

用法：
  python evaluate.py                              # 默认参数
  python evaluate.py --key YOUR_KEY               # 指定 API Key
  python evaluate.py --no-llm-judge               # 跳过 LLM 裁判（更快）
  python evaluate.py --file test/my_cases.json    # 自定义测试集
  python evaluate.py --threshold 0.8              # 调高"正确"判断阈值

输出：
  test/evaluation_report.json   详细报告
  控制台打印汇总表格

与旧版的区别：
  旧版：每条问答需要人工输入 y/n
  新版：完全自动，无任何人工干预
  新版：5 项量化指标（语义相似度 / LLM 裁判 / 忠实度 / 相关度 / 耗时百分位）
"""
import json
import os
import sys
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

# 让脚本在项目根目录运行时能找到 src 包
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_TEST_FILE   = "test/test_cases.json"
DEFAULT_REPORT_PATH = "test/evaluation_report.json"
API_BASE_URL        = os.environ.get("API_BASE_URL", "http://localhost:8000")


def load_test_cases(file_path: str) -> list:
    """加载测试用例，不存在时生成示例文件"""
    p = Path(file_path)
    if not p.exists():
        print(f"⚠️  未找到测试文件 {file_path}，正在生成示例模板...")
        p.parent.mkdir(parents=True, exist_ok=True)
        example = [
            {
                "question": "这篇文献的主要研究方法是什么？",
                "ground_truth": "文献采用了定量分析与案例研究相结合的混合方法。",
                "category": "方法论"
            },
            {
                "question": "文中提出了哪些核心论点？",
                "ground_truth": "文中提出了三个核心论点：效率提升、成本降低、用户体验改善。",
                "category": "核心论点"
            },
        ]
        p.write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅  示例文件已创建：{file_path}")
    return json.loads(p.read_text(encoding="utf-8"))


def call_ask(question: str, api_key: str, max_retries: int = 3) -> tuple[str, list, float]:
    """
    调用 /ask 接口，返回 (answer, sources, latency_ms)。

    带重试：评估会短时间内连发几十个请求，容易触发 LLM 服务端限流，
    导致后端返回 500 / 空答案。这类瞬时错误会污染评估结果（把一个
    本可正确回答的问题记为空答案），因此对 5xx、连接异常、空答案做
    指数退避重试。真实的"知识库中暂无此信息"是非空文本，不会被误重试。

    所有重试都失败时返回空答案，不抛异常（让评估继续跑完其余用例）。
    """
    headers = {"x-api-key": api_key}
    last_err = ""
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_BASE_URL}/ask",
                json={"question": question},
                headers=headers,
                timeout=180,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                if answer.strip():
                    return answer, data.get("sources", []), latency_ms
                last_err = "空答案"
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:80]}"
            else:
                # 4xx（鉴权、参数错误等）重试也没用，直接返回
                print(f"  ⚠️  HTTP {resp.status_code}: {resp.text[:100]}")
                return "", [], latency_ms
        except Exception as e:
            last_err = str(e)

        if attempt < max_retries:
            backoff = 2.0 * attempt
            print(f"  ⏳  第 {attempt} 次失败（{last_err[:60]}），{backoff:.0f}s 后重试…")
            time.sleep(backoff)

    print(f"  ❌  重试 {max_retries} 次仍失败: {last_err[:80]}")
    return "", [], (time.perf_counter() - t0) * 1000


def build_evaluator(api_key: str, enable_llm: bool):
    """
    初始化 AutoEvaluator。
    Embedding 模型直接从本地加载（不走 API），LLM 可选。
    """
    from src.config import settings
    from src.evaluator import AutoEvaluator

    print("正在加载 Embedding 模型...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        local_path = os.path.abspath(settings.EMBEDDING_MODEL_PATH)
        embeddings = HuggingFaceEmbeddings(
            model_name=local_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"  ✅  Embedding 模型加载完成: {local_path}")
    except Exception as e:
        print(f"  ❌  Embedding 加载失败，语义评分将为 0: {e}")
        embeddings = None

    llm = None
    if enable_llm and embeddings is not None:
        print("正在初始化 LLM 裁判...")
        try:
            from src.llm_factory import get_llm_with_fallback
            llm = get_llm_with_fallback()
            print("  ✅  LLM 裁判已就绪")
        except Exception as e:
            print(f"  ⚠️  LLM 初始化失败，跳过 LLM 裁判和忠实度评分: {e}")

    return AutoEvaluator(embeddings=embeddings, llm=llm)


def print_summary(summary: dict) -> None:
    """控制台打印汇总表格"""
    print("\n" + "=" * 60)
    print("📊  评估结果汇总")
    print("=" * 60)
    print(f"  测试用例总数    : {summary['total']}")
    print(f"  正确数          : {summary['correct']}")
    print(f"  准确率          : {summary['accuracy'] * 100:.1f}%  (阈值 ≥ {summary['correct_threshold']})")
    print(f"  平均语义相似度  : {summary['avg_semantic']:.4f}")
    print(f"  平均答案相关度  : {summary['avg_relevance']:.4f}")
    if summary.get("avg_llm_judge") is not None:
        print(f"  平均 LLM 裁判分 : {summary['avg_llm_judge']:.4f}  (满分 1.0)")
    if summary.get("avg_faithfulness") is not None:
        print(f"  平均忠实度      : {summary['avg_faithfulness']:.4f}")
    print(f"  平均耗时        : {summary['avg_latency_ms']:.0f} ms")
    print(f"  P50 耗时        : {summary['p50_latency_ms']:.0f} ms")
    print(f"  P90 耗时        : {summary['p90_latency_ms']:.0f} ms")
    print(f"  P99 耗时        : {summary['p99_latency_ms']:.0f} ms")

    if summary.get("category_scores"):
        print("\n  分类语义得分：")
        for cat, score in summary["category_scores"].items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"    {cat:<15} {bar} {score:.4f}")
    print("=" * 60)


def run(args) -> None:
    api_key = args.key or os.environ.get("API_KEY", "")
    if not api_key:
        print("❌  未提供 API Key，请用 --key 参数或设置 API_KEY 环境变量")
        sys.exit(1)

    test_cases = load_test_cases(args.file)
    print(f"\n🚀  开始评估 | 共 {len(test_cases)} 条测试用例")
    print(f"    API: {API_BASE_URL}")
    print(f"    LLM 裁判: {'开启' if not args.no_llm_judge else '关闭'}")
    print(f"    正确阈值: {args.threshold}")
    print()

    evaluator = build_evaluator(api_key, enable_llm=not args.no_llm_judge)

    results = []
    for i, case in enumerate(test_cases, 1):
        question    = case["question"]
        ground_truth = case.get("ground_truth", "")
        category    = case.get("category", "未分类")

        print(f"[{i:2d}/{len(test_cases)}] {category} | {question[:60]}")

        answer, sources, latency_ms = call_ask(question, api_key)
        print(f"        耗时: {latency_ms:.0f}ms | 来源: {len(sources)} 条")

        # 请求间隔：避免短时间高频请求触发 LLM 服务端限流
        if i < len(test_cases):
            time.sleep(args.delay)

        if evaluator._embeddings is None:
            # 降级：无 Embedding 时只记录耗时
            scored = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "category": category,
                "latency_ms": round(latency_ms, 1),
                "semantic_score": 0.0,
                "answer_relevance": 0.0,
                "llm_judge_score": -1,
                "llm_judge_reason": "Embedding 不可用",
                "faithfulness": None,
                "is_correct": False,
                "sources_count": len(sources),
            }
        else:
            evaluator._threshold = args.threshold
            scored = evaluator.score_one(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                sources=sources,
                latency_ms=latency_ms,
                category=category,
                enable_llm_judge=not args.no_llm_judge,
                enable_faithfulness=not args.no_llm_judge,
            )
        results.append(scored)
        sem = scored["semantic_score"]
        ok  = "✅" if scored["is_correct"] else "❌"
        print(f"        {ok} 语义: {sem:.3f}"
              + (f" | LLM裁判: {scored['llm_judge_score']}/3" if scored["llm_judge_score"] >= 0 else "")
              + (f" | 忠实度: {scored['faithfulness']:.2f}" if scored.get("faithfulness") is not None else ""))

    summary = evaluator.summarize(results)
    summary["test_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 保存报告
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "details": results}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary)
    print(f"\n📁  详细报告已保存至: {report_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全自动 RAG 评估工具")
    parser.add_argument("--api",          default=API_BASE_URL,        help="API 地址")
    parser.add_argument("--file",         default=DEFAULT_TEST_FILE,   help="测试用例 JSON 文件路径")
    parser.add_argument("--report",       default=DEFAULT_REPORT_PATH, help="报告输出路径")
    parser.add_argument("--key",          default="",                  help="API Key")
    parser.add_argument("--threshold",    type=float, default=0.75,    help="语义相似度正确阈值（默认 0.75）")
    parser.add_argument("--no-llm-judge", action="store_true",         help="跳过 LLM 裁判和忠实度评分（更快）")
    parser.add_argument("--delay",        type=float, default=1.5,     help="每条用例之间的请求间隔秒数，降低限流概率（默认 1.5）")
    run(parser.parse_args())
