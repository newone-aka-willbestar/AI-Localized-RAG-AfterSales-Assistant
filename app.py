import streamlit as st
import requests
import json
import os
import uuid

st.set_page_config(page_title="华科制造智能售后客服", page_icon="🤖", layout="wide")

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_API_KEY = os.environ.get("API_KEY", "")
REPORT_PATH = "test/evaluation_report.json"

# --- 侧边栏 ---
with st.sidebar:
    st.title("🛠️ 管理后台")
    menu = st.radio("选择功能", ["智能客服对话", "系统评估看板"])
    st.divider()

    st.subheader("系统设置")
    api_key = st.text_input("API Key", value=DEFAULT_API_KEY, type="password")

    # --- PDF 上传 ---
    st.subheader("📄 上传 PDF 文档")
    uploaded_file = st.file_uploader("选择 PDF 文件", type=["pdf"])
    if uploaded_file and st.button("开始向量化上传", type="primary"):
        with st.spinner("正在解析文档并构建索引..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            headers = {"x-api-key": api_key}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ 成功入库 {data.get('total_chunks', '?')} 个文本块")
                else:
                    st.error(f"处理失败: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"连接 API 失败: {e}")

    st.divider()

    # --- 网页抓取 ---
    st.subheader("🌐 抓取网页内容")
    url_input = st.text_area(
        "输入网页 URL（每行一个）",
        placeholder="https://example.com/manual\nhttps://example.com/faq",
        height=100,
    )
    force_recrawl = st.checkbox("强制重新抓取（忽略去重）", value=False)
    if st.button("开始抓取并入库"):
        urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
        if not urls:
            st.warning("请先输入至少一个 URL")
        else:
            with st.spinner(f"正在抓取 {len(urls)} 个网页..."):
                headers = {"x-api-key": api_key, "Content-Type": "application/json"}
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/crawl",
                        json={"urls": urls, "force": force_recrawl, "translate": False},
                        headers=headers,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        succeeded = data.get("succeeded", [])
                        skipped = data.get("skipped", [])
                        failed = data.get("failed", [])
                        if succeeded:
                            st.success(f"✅ 成功入库 {len(succeeded)} 个网页")
                            for item in succeeded:
                                st.caption(f"  · {item['url']} → {item['chunks']} 块")
                        if skipped:
                            st.info(f"⏭️ {len(skipped)} 个 URL 已抓取过，跳过")
                        if failed:
                            st.error(f"❌ {len(failed)} 个抓取失败")
                            for item in failed:
                                st.caption(f"  · {item['url']}: {item['error']}")
                    else:
                        st.error(f"请求失败: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"连接 API 失败: {e}")


# --- 页面 A：智能客服对话 ---
if menu == "智能客服对话":
    st.title("🤖 华科制造 AI 智能售后")
    st.caption("基于 RAG 混合检索 · 实时参考技术手册回答问题")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    # session_id 在整个浏览器会话中保持不变，用于服务端多轮记忆
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔍 查看检索到的原文片段"):
                    for src in msg["sources"]:
                        st.info(
                            f"📄 来源: {src.get('source', '未知')}\n\n"
                            f"{src.get('content_excerpt', '...')}"
                        )

    # 清空对话按钮：同时清除前端消息历史和服务端会话记忆
    col_input, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🗑️ 清空对话", use_container_width=True):
            # 通知服务端释放该 session 的记忆
            try:
                requests.delete(
                    f"{API_BASE_URL}/session/{st.session_state.session_id}",
                    headers={"x-api-key": api_key},
                    timeout=5,
                )
            except Exception:
                pass  # 网络异常不影响前端清空
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())  # 新会话 ID
            st.rerun()

    if prompt := st.chat_input("请描述设备故障或查询参数..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在检索手册并生成回答..."):
                try:
                    headers = {"x-api-key": api_key}
                    resp = requests.post(
                        f"{API_BASE_URL}/ask",
                        json={"question": prompt, "session_id": st.session_state.session_id},
                        headers=headers,
                        timeout=180,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "未能生成答案")
                        sources = data.get("sources", [])
                        intent = data.get("intent", "")
                        st.markdown(answer)
                        if intent:
                            st.caption(f"意图识别：{intent}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "intent": intent,
                            "question": prompt,
                        })
                    else:
                        st.error(f"API 响应异常: {resp.status_code}")
                except Exception as e:
                    st.error(f"发生错误: {e}")


# --- 页面 B：评估看板 ---
elif menu == "系统评估看板":
    st.title("📊 系统性能与准确率评估")

    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                report = json.load(f)

            summary = report.get("summary", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("端到端准确率", summary.get("accuracy", "N/A"))
            c2.metric("平均响应耗时", f"{summary.get('avg_latency_sec', 0)}s")
            c3.metric("测试用例总数", summary.get("total_questions", 0))

            st.divider()
            st.subheader("详细评测流水")
            for item in report.get("details", []):
                status = "✅ 通过" if item["is_correct"] else "❌ 失败"
                with st.expander(f"{status} | {item['question']}"):
                    st.write(f"**AI 回答:** {item['answer']}")
                    st.write(f"**耗时:** {item['latency']}s")
                    if item.get("sources"):
                        st.json(item["sources"])
        except Exception as e:
            st.error(f"读取报告出错: {e}")
    else:
        st.warning("⚠️ 暂无评估报告")
        st.info("运行以下命令生成报告：")
        st.code("python evaluate.py --key your-secret-key-2026")
