import streamlit as st
import requests
import json
import os
from pathlib import Path

st.set_page_config(page_title="华科制造智能售后客服", page_icon="🤖", layout="wide")

API_BASE_URL = "http://localhost:8000"
REPORT_PATH = "test/evaluation_report.json"

# --- 侧边栏：功能切换 ---
with st.sidebar:
    st.title("🛠️ 管理后台")
    menu = st.radio("选择功能", ["智能客服对话", "系统评估看板"])
    st.divider()
    
    st.subheader("系统设置")
    api_key = st.text_input("API Key", value="your-secret-key-2026", type="password")
    
    st.subheader("文档管理")
    uploaded_file = st.file_uploader("上传产品手册 / PDF", type=["pdf"])
    if uploaded_file and st.button("开始向量化上传", type="primary"):
        with st.spinner("正在解析文档并构建索引..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            headers = {"X-API-Key": api_key}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers)
                if resp.status_code == 200:
                    st.success(f"✅ {uploaded_file.name} 已加入知识库")
                else:
                    st.error(f"处理失败: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"连接 API 失败: {e}")

# --- 逻辑 A：智能客服对话界面 ---
if menu == "智能客服对话":
    st.title("🤖 华科制造 AI 智能售后")
    st.caption("基于 RAG 混合检索技术 · 实时参考技术手册回答问题")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 渲染历史消息
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔍 查看检索到的原文片段"):
                    for src in msg["sources"]:
                        st.info(f"📄 来源: {src.get('source', '未知')} (第 {src.get('page', '?')} 页)\n\n内容: {src.get('content_excerpt', '...')}")

    # 用户输入
    if prompt := st.chat_input("请描述设备故障或查询参数..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索手册并生成回答..."):
                try:
                    headers = {"X-API-Key": api_key}
                    resp = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt},
                        headers=headers,
                        timeout=180
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "未能生成答案")
                        sources = data.get("sources", [])
                        
                        st.markdown(answer)
                        # 面试加分点：反馈按钮
                        col1, col2 = st.columns([1, 10])
                        with col1: st.button("👍", key=f"up_{len(st.session_state.messages)}")
                        with col2: st.button("👎", key=f"down_{len(st.session_state.messages)}")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error("API 响应异常")
                except Exception as e:
                    st.error(f"发生错误: {e}")

# --- 逻辑 B：系统评估看板 (这就是你的量化成果展示) ---
elif menu == "系统评估看板":
    st.title("📊 系统性能与准确率评估")
    
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        summary = report.get("summary", {})
        
        # 核心指标显示
        c1, c2, c3 = st.columns(3)
        c1.metric("端到端准确率", summary.get("accuracy", "N/A"))
        c2.metric("平均响应耗时", f"{summary.get('avg_latency_sec', 0)}s")
        c3.metric("测试用例总数", summary.get("total_questions", 0))
        
        st.divider()
        st.subheader("详细评测流水")
        
        # 将详情转为表格
        details = report.get("details", [])
        for item in details:
            status = "✅ 通过" if item["is_correct"] else "❌ 失败"
            with st.expander(f"{status} | {item['question']}"):
                st.write(f"**AI 回答:** {item['answer']}")
                st.write(f"**耗时:** {item['latency']}s")
                if item.get("sources"):
                    st.json(item["sources"])
    else:
        st.warning("暂无评估报告，请先运行 `python test/evaluate.py` 进行系统测算。")
        st.info("面试建议：在本地运行评估脚本后，该页面将展示你的系统优化成果。")