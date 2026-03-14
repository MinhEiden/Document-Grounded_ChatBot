import streamlit as st
import os
from dotenv import load_dotenv
from retrieval_pipeline.retriever import retrieve_context
from retrieval_pipeline.generator import generate_answer
from retrieval_pipeline.query_rewriter import rewrite_query
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings

load_dotenv()

# Kiểm tra các API Key bắt buộc (Bỏ OPENAI_API_KEY vì dùng Ollama)
if not os.getenv("COHERE_API_KEY"):
    st.warning("⚠️ Cảnh báo: Bạn cần cấu hình COHERE_API_KEY trong file .env để hệ thống vector hóa hoạt động.")

def get_ingested_files():
    """Lấy danh sách các file đã được ingest vào ChromaDB."""
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    if not os.path.exists(chroma_path):
        return []
    
    try:
        embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
        db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
        data = db.get()
        if not data or "metadatas" not in data or not data["metadatas"]:
            return []
            
        sources = set()
        for meta in data["metadatas"]:
            if meta and "source" in meta:
                # Chỉ lấy tên file từ đường dẫn
                sources.add(os.path.basename(meta["source"]))
        return sorted(list(sources))
    except Exception as e:
        return []

st.set_page_config(page_title="Academic Chatbot", page_icon="📚", layout="wide")
st.title("📚 Academic Chatbot")
st.caption("Hỏi đáp tài liệu học thuật với AI")

# Sidebar - Hiển thị trạng thái dữ liệu
with st.sidebar:
    st.header("📁 Trạng thái dữ liệu")
    
    ingested_files = get_ingested_files()
    
    if not ingested_files:
        st.info("Chưa có file nào trong cuộc trò chuyện.")
    else:
        st.success(f"Đã nạp {len(ingested_files)} tài liệu vào hệ thống:")
        for f in ingested_files:
            st.markdown(f"- 📄 {f}")
            
    st.divider()
    st.caption("💡 *Lưu ý: Quá trình phân tích và nạp tài liệu (Ingestion) hiện được thực hiện thông qua dòng lệnh.*")
    st.code("python -c 'from ingestion_pipeline import run_ingestion_pipeline; run_ingestion_pipeline()'", language="bash")

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
            
            # Lấy list of documents thay vì string thuần
            retrieved_docs = retrieve_context(rewritten)
            
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
