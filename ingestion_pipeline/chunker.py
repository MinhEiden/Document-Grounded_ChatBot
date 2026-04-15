import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def process_and_enrich_documents(documents: list) -> list:
    """Chia tài liệu dùng Semantic Chunker kết hợp tóm tắt từ LLM để tạo Fusion Chunks."""
    print("Đang tải model HuggingFace embedding (Local) cho Semantic Chunking...")
    local_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    print(" Đang tiến hành semantic chunking (chia theo ngữ nghĩa)...")
    semantic_chunker = SemanticChunker(
        local_embeddings,
        breakpoint_threshold_type="percentile"
    )
    
    chunks = semantic_chunker.split_documents(documents)
    print(f"✅ Đã tạo ra {len(chunks)} semantic chunks từ {len(documents)} tài liệu đầu vào.")
    
    print("Khởi tạo LLM ChatOllama (llama3.2) để trích xuất tóm tắt...")
    llm = ChatOllama(model="llama3.2", temperature=0.1)
    
    # Tạo prompt đóng vai trò hội thoại ảo ép Llama 3.2 tóm tắt siêu ngắn (< 40 từ)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý ảo chuyên trích xuất thông tin. Hãy tóm tắt nội dung cốt lõi của đoạn văn bản được cung cấp với tối đa 40 từ. Chỉ trả lời bằng nội dung tóm tắt, không thêm bất kỳ nhận xét, giải thích hay lời chào nào."),
        ("human", "Hãy tóm tắt đoạn văn bản sau đây, tập trung vào những khái niệm cốt lõi nhất (tối đa 40 từ):\n\n{text}")
    ])
    
    chain = prompt | llm
    
    enriched_chunks = []
    # Sử dụng tqdm để hiển thị thanh tiến trình trong terminal
    for chunk in tqdm(chunks, desc="Enriching Documents", unit="chunk"):
        # Lấy tên file gốc (tránh path dài thòng)
        filepath = chunk.metadata.get("source", "Unknown")
        filename = os.path.basename(filepath)
        
        # Lưu nội dung gốc
        original_text = chunk.page_content
        
        # Invoke chuỗi suy luận bằng LLM
        summary_response = chain.invoke({"text": original_text})
        summary = summary_response.content.strip()
        
        # Trộn dữ liệu thành "Fusion Chunk" format
        fusion_content = f"[Tài liệu: {filename}]\n[Tóm tắt: {summary}]\n[Nội dung: {original_text}]"
        
        enriched_doc = Document(
            page_content=fusion_content,
            metadata=chunk.metadata
        )
        enriched_chunks.append(enriched_doc)
        
    return enriched_chunks
