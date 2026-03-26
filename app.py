import streamlit as st
import requests

st.set_page_config(page_title="华科制造智能售后客服", page_icon="🤖", layout="wide")

API_BASE_URL = "http://localhost:8000"

st.title("华科制造智能售后客服系统")
st.markdown("基于 RAG 的本地智能客服 · 支持文档上传")

# 侧边栏
with st.sidebar:
    st.subheader("系统设置")
    api_key = st.text_input("API Key", value="your-secret-key-2026", type="password")
    st.divider()
    
    st.subheader("文档上传")
    uploaded_file = st.file_uploader("上传产品手册 / PDF", type=["pdf"])
    
    if uploaded_file and st.button("上传到知识库", type="primary"):
        with st.spinner("正在上传并向量化..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            headers = {"X-API-Key": api_key}
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, headers=headers)
                if resp.status_code == 200:
                    st.success(f"✅ {uploaded_file.name} 上传成功！")
                else:
                    st.error(f"上传失败: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"上传异常: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 参考来源"):
                sources = msg["sources"]
                if isinstance(sources, list):
                    for i, src in enumerate(sources, 1):
                        if isinstance(src, dict):
                            source_name = src.get("metadata", {}).get("source", "未知文档")
                        else:
                            source_name = str(src)
                        st.write(f"**{i}.** {source_name}")

# 用户输入
if prompt := st.chat_input("请输入您的问题，例如：这个设备的保修期是多久？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤖 正在思考中...（可能需要 30~90 秒，请耐心等待）"):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"question": prompt},
                    headers={"X-API-Key": api_key},
                    timeout=180          # ← 关键修复：从 60 秒改成 180 秒
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "抱歉，我暂时无法回答。")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"请求失败: {resp.text}")
            except Exception as e:
                st.error(f"发生错误: {e}")

st.caption("提示：第一次提问前请先上传 PDF 文档")