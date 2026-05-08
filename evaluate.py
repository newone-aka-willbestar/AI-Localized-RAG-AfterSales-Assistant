import json
import os
import requests
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# 配置测试案例路径
DEFAULT_TEST_FILE = "test/test_cases.json"
DEFAULT_REPORT_PATH = "test/evaluation_report.json"

# 从环境变量读取 API Key，不硬编码
# 本地运行：在 .env 里设置 API_KEY=xxx
# 也可以通过命令行参数传入：python evaluate.py --key xxx
API_KEY = os.environ.get("API_KEY", "")

def load_test_cases(file_path: str):
    """加载测试用例"""
    if not Path(file_path).exists():
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
    """RAG 系统评估工具"""
    test_cases = load_test_cases(test_file)
    results = []
    total_latency = 0
    correct_count = 0

    print("\n" + "="*50)
    print("🚀 华科制造 AI 智能客服 - RAG 系统专业评估开始")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试集规模: {len(test_cases)} 条真实业务问题")
    print("="*50 + "\n")

    # 构造请求头
    headers = {"X-API-Key": API_KEY}

    for i, case in enumerate(test_cases):
        question = case["question"]
        category = case.get("category", "通用")
        
        print(f"测试 [{i+1}/{len(test_cases)}] | 类别: {category}")
        print(f"❓ 问题: {question}")
        
        start_time = time.time()
        
        try:
            # --- 修复点：加入 headers 验证 ---
            resp = requests.post(
                api_url, 
                json={"question": question}, 
                headers=headers, 
                timeout=180
            )
            latency = time.time() - start_time
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "无返回内容")
                sources = data.get("sources", [])
            else:
                answer = f"HTTP错误: {resp.status_code} - {resp.text}"
                sources = []
                latency = 0
        except Exception as e:
            answer = f"请求异常: {str(e)}"
            sources = []
            latency = 0

        total_latency += latency
        
        print(f"🤖 AI回答: {answer[:300]}..." if len(answer) > 300 else f"🤖 AI回答: {answer}")
        print(f"⏱️ 响应耗时: {latency:.2f}s")
        
        is_ok = input("✅ 是否准确、基于文档且专业？(y/n): ").strip().lower()
        is_correct = (is_ok == 'y')
        
        if is_correct:
            correct_count += 1

        results.append({
            "question": question,
            "category": category,
            "answer": answer,
            "latency": round(latency, 2),
            "is_correct": is_correct,
            "sources": sources
        })
        print("-" * 30)

    accuracy = (correct_count / len(test_cases)) * 100 if test_cases else 0
    avg_latency = total_latency / len(test_cases) if test_cases else 0

    print("\n" + "📊" + " 评估报告摘要 " + "📊")
    print(f"🎯 端到端准确率: {accuracy:.1f}%")
    print(f"⚡ 平均响应耗时: {avg_latency:.2f}s")
    print(f"📝 详细结果已保存至: {DEFAULT_REPORT_PATH}")
    print("="*50)

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
    parser = argparse.ArgumentParser(description="RAG 系统评估工具")
    parser.add_argument("--api", default="http://localhost:8000/ask", help="FastAPI 地址")
    parser.add_argument("--file", default=DEFAULT_TEST_FILE, help="测试用例 JSON 文件路径")
    parser.add_argument("--key", default="", help="API Key（优先级高于环境变量）")

    args = parser.parse_args()

    # 命令行参数优先级 > 环境变量
    if args.key:
        API_KEY = args.key

    try:
        evaluate(args.api, args.file)
    except KeyboardInterrupt:
        print("\n👋 评估被用户中断。")