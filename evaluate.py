import json
import requests
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# 配置测试案例路径
DEFAULT_TEST_FILE = "test/test_cases.json"
DEFAULT_REPORT_PATH = "test/evaluation_report.json"

def load_test_cases(file_path: str):
    """加载测试用例"""
    if not Path(file_path).exists():
        # 如果文件不存在，自动创建一个示例模板
        print(f"⚠️ 未找到测试文件 {file_path}，正在创建示例模板...")
        example_cases = [
            {"question": "设备的保修期是多久？", "category": "政策类"},
            {"question": "错误代码 E05 怎么处理？", "category": "故障类"}
        ]
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(example_cases, f, ensure_ascii=False, indent=2)
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(api_url: str, test_file: str):
    """
    RAG 系统评估工具：测试准确率、响应耗时、数据分布
    """
    test_cases = load_test_cases(test_file)
    results = []
    total_latency = 0
    correct_count = 0

    print("\n" + "="*50)
    print("🚀 华科制造 AI 智能客服 - RAG 系统专业评估开始")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"集规模: {len(test_cases)} 条真实业务问题")
    print("="*50 + "\n")

    for i, case in enumerate(test_cases):
        question = case["question"]
        category = case.get("category", "通用")
        
        print(f"测试 [{i+1}/{len(test_cases)}] | 类别: {category}")
        print(f"❓ 问题: {question}")
        
        # 1. 记录耗时开始
        start_time = time.time()
        
        try:
            # 2. 调用 API
            resp = requests.post(api_url, json={"question": question}, timeout=180)
            latency = time.time() - start_time # 耗时计算
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "无返回内容")
                sources = data.get("sources", [])
            else:
                answer = f"HTTP错误: {resp.status_code}"
                sources = []
                latency = 0
        except Exception as e:
            answer = f"请求异常: {str(e)}"
            sources = []
            latency = 0

        # 3. 统计数据
        total_latency += latency
        
        # 4. 人工交互判断 (中小厂最看重的负责任态度)
        print(f"🤖 AI回答: {answer[:300]}..." if len(answer) > 300 else f"🤖 AI回答: {answer}")
        print(f"⏱️ 响应耗时: {latency:.2f}s")
        
        is_ok = input("✅ 是否准确、基于文档且专业？(y/n): ").strip().lower()
        is_correct = (is_ok == 'y')
        
        if is_correct:
            correct_count += 1

        # 5. 保存单条详细结果
        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "latency": round(latency, 2),
            "is_correct": is_correct,
            "sources": sources
        })
        print("-" * 30)

    # 6. 计算最终指标
    accuracy = (correct_count / len(test_cases)) * 100 if test_cases else 0
    avg_latency = total_latency / len(test_cases) if test_cases else 0

    # 7. 打印评估报告摘要
    print("\n" + "📊" + " 评估报告摘要 " + "📊")
    print(f"🎯 端到端准确率: {accuracy:.1f}%")
    print(f"⚡ 平均响应耗时: {avg_latency:.2f}s")
    print(f"📝 详细结果已保存至: {DEFAULT_REPORT_PATH}")
    print("="*50)

    # 8. 保存 JSON 报告
    final_report = {
        "summary": {
            "test_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_questions": len(test_cases),
            "accuracy": f"{accuracy:.1f}%",
            "avg_latency_sec": round(avg_latency, 2)
        },
        "details": results
    }
    
    with open(DEFAULT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="应届生项目 RAG 评估工具")
    parser.add_argument("--api", default="http://localhost:8000/ask", help="FastAPI 地址")
    parser.add_argument("--file", default=DEFAULT_TEST_FILE, help="测试用例 JSON 文件路径")
    
    args = parser.parse_args()
    
    try:
        evaluate(args.api, args.file)
    except KeyboardInterrupt:
        print("\n👋 评估被用户中断。")