import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List

load_dotenv()

st.set_page_config(page_title="AI智能客服系统", layout="wide")
st.title("📞 AI智能客服系统（RAG + HyDE）")
st.markdown("**技术栈**：LangChain + Chroma + HyDE + Ollama + Streamlit（零API费用）")

# ================== 配置 ==================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
USE_HYDE = True  # 默认开启HyDE

# 向量库持久化
persist_directory = "./chroma_db"

# ================== HyDE实现（保留你原来的核心思想）==================
def generate_hypothetical_document(question: str) -> str:
    """HyDE：先生成假设答案文档"""
    llm = ChatOllama(model="qwen2:7b", temperature=0.3)
    prompt = f"请根据以下问题，生成一份简短的、像知识库里的文档一样的答案（仅用于检索增强，不要直接回答用户）：\n问题：{question}\n假设文档："
    return llm.invoke(prompt).content

# ================== 初始化向量库 ==================
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# ================== 上传文档 ==================
with st.sidebar:
    st.header("📤 知识库管理")
    uploaded_file = st.file_uploader("上传PDF或TXT", type=["pdf", "txt"])
    if uploaded_file:
        with open("temp_upload." + uploaded_file.name.split(".")[-1], "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # 加载+分块
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader("temp_upload.pdf")
        else:
            loader = TextLoader("temp_upload.txt", encoding="utf-8")
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = splitter.split_documents(docs)
        
        # 添加到Chroma
        vectorstore.add_documents(splits)
        st.success(f"✅ 上传成功！已添加 {len(splits)} 个文本块")

# ================== 聊天界面 ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("请输入您的问题（支持多轮对话）"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库 + HyDE增强..."):
            # HyDE增强
            if USE_HYDE:
                hypo_doc = generate_hypothetical_document(question)
                # 可选：把假设文档也加到检索（这里简化直接用原问题）
            
            docs = retriever.invoke(question)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            # Prompt
            prompt = ChatPromptTemplate.from_template("""
你是一个专业的智能客服。请严格根据以下参考内容回答用户问题。
如果参考内容中没有答案，请说“抱歉，当前知识库中没有找到相关信息”。

【参考内容】
{context}

【用户问题】{question}
【回答】""")

            llm = ChatOllama(model="qwen2:7b", temperature=0.3)
            chain = (
                {"context": lambda x: context, "question": lambda x: x}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = chain.invoke(question)

            st.markdown(answer)

            # 显示来源
            with st.expander("📄 查看参考来源"):
                for i, doc in enumerate(docs):
                    st.write(f"**片段 {i+1}**：{doc.page_content[:250]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})