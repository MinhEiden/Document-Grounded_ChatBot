import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

from ingestion_pipeline.orchestrator import ingest_file
from retrieval_pipeline.retriever import retrieve_context
from retrieval_pipeline.generator import generate_answer
from retrieval_pipeline.query_rewriter import rewrite_query
from utils.config import get_config
from utils.file_handler import save_uploaded_file
from utils.session_manager import get_or_create_session_id

load_dotenv()

# Kiểm tra các API Key bắt buộc (Bỏ OPENAI_API_KEY vì dùng Ollama)
if not get_config("COHERE_API_KEY"):
    st.warning("⚠️ Cảnh báo: Bạn cần cấu hình COHERE_API_KEY trong file .env để hệ thống vector hóa hoạt động.")

def get_ingested_files(session_id: str):
    """Lấy danh sách file đã ingest cho session hiện tại."""
    chroma_path = get_config("CHROMA_DB_PATH", "./vector_store")
    if not os.path.exists(chroma_path):
        return []

    try:
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings,
            collection_name="academic_chatbot",
        )
        data = db.get(where={"session_id": session_id})
        if not data or "metadatas" not in data or not data["metadatas"]:
            return []

        sources = set()
        for meta in data["metadatas"]:
            if meta:
                filename = meta.get("filename") or os.path.basename(meta.get("source", ""))
                if filename:
                    sources.add(filename)
        return sorted(list(sources))
    except Exception:
        return []

st.set_page_config(page_title="Academic Chatbot", page_icon="📚", layout="wide")
session_id = get_or_create_session_id(st.session_state)

st.title("📚 Academic Chatbot")
st.caption("Hỏi đáp tài liệu học thuật với AI")

# Sidebar - Hiển thị trạng thái dữ liệu
with st.sidebar:
    st.header("📁 Trạng thái dữ liệu")

    uploaded_files = st.file_uploader(
        "Kéo/thả hoặc chọn file (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest tài liệu", use_container_width=True):
        ingest_results = []
        for uf in uploaded_files:
            saved_path = save_uploaded_file(uf, session_id=session_id)
            info = ingest_file(saved_path, session_id=session_id)
            if info:
                ingest_results.append((uf.name, info))

        if ingest_results:
            st.success(f"Đã ingest {len(ingest_results)} file cho phiên hiện tại.")
            for name, info in ingest_results:
                st.markdown(f"- ✅ {name} (chunks: {info['chunk_count']}, file_id: {info['file_id']})")
        else:
            st.error("Ingest thất bại. Kiểm tra lại định dạng hoặc log.")

    ingested_files = get_ingested_files(session_id)
    if not ingested_files:
        st.info("Chưa có file nào trong phiên này.")
    else:
        st.success(f"Đã nạp {len(ingested_files)} tài liệu trong phiên:")
        for f in ingested_files:
            st.markdown(f"- 📄 {f}")

    st.caption(f"Session ID: {session_id}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "docs" in msg and msg["docs"]:
            with st.expander("📚 Tài liệu tham khảo"):
                for i, doc in enumerate(msg["docs"]):
                    source = doc.metadata.get("source", "Không rõ nguồn")
                    filename = os.path.basename(source)
                    st.markdown(f"**Chunk {i+1}** - *{filename}*")
                    st.info(doc.page_content)

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Lấy 10 tin nhắn gần nhất làm buffer (5 user, 5 assistant)
    chat_history_buffer = st.session_state.messages[-10:] if len(st.session_state.messages) > 0 else []

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            # Đưa history vào để rewrite tạo Standalone Question
            rewritten = rewrite_query(prompt, chat_history_buffer)
            # Có thể in mờ ra để debug xem nó có viết lại đúng không
            # st.caption(f"*Standalone query:* {rewritten}")
            
            retrieved_docs = retrieve_context(rewritten, session_id=session_id)
            
            # Khởi tạo giá trị ban đầu để tạo answer
            if not retrieved_docs:
                context_str = ""
            else:
                context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                
            answer = generate_answer(rewritten, context_str)
            
        st.markdown(answer)
        
        # Gắn thêm tài liệu tham khảo nếu có trong một expander
        if retrieved_docs:
            with st.expander("📚 Tài liệu tham khảo"):
                for i, doc in enumerate(retrieved_docs):
                    # Lấy metadata source, nếu không có thì gán 'Không rõ nguồn'
                    source = doc.metadata.get("source", "Không rõ nguồn")
                    filename = os.path.basename(source)
                    st.markdown(f"**Chunk {i+1}** - *{filename}*")
                    # In toàn bộ nội dung của chunk hoặc giới hạn tùy bạn, ở đây in ra với blockquote
                    st.info(doc.page_content)
                    
    st.session_state.messages.append({"role": "assistant", "content": answer, "docs": retrieved_docs})
