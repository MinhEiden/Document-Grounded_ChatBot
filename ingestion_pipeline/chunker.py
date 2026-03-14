from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def chunk_documents(documents: list) -> list:
    """Chia tài liệu dùng Semantic Chunker kết hợp Local Embeddings để tiết kiệm request."""
    print("⏳ Đang tải model HuggingFace embedding (Local) cho Semantic Chunking...")
    # Dùng model nhẹ, nhanh, hỗ trợ đa ngữ. Nếu muốn chuyên Tiếng Việt, có thể thử 'keepitreal/vietnamese-sbert'
    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("⏳ Đang tiến hành semantic chunking (chia theo ngữ nghĩa)...")
    semantic_chunker = SemanticChunker(
        local_embeddings,
        breakpoint_threshold_type="percentile" # Bạn có thể đổi sang 'standard_deviation' nếu muốn
    )
    
    chunks = semantic_chunker.split_documents(documents)
    print(f"✅ Xong! Đã tạo ra {len(chunks)} semantic chunks từ {len(documents)} tài liệu đầu vào.")
    return chunks
