import os
import time
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma


def embed_and_store(chunks: list):
    """Tạo embedding bằng Cohere và lưu vào ChromaDB, chia batch để tránh rate limit."""
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    print(f"⏳ Đang nhúng tổng cộng {len(chunks)} chunks bằng Cohere...")
    
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
    
    vector_store = Chroma(
        persist_directory=chroma_path, 
        embedding_function=embeddings
    )
    
    batch_size = 25
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        current_batch = (i // batch_size) + 1
        
        print(f"🚀 Vector hóa & Lưu DB batch {current_batch}/{total_batches} ({len(batch)} chunks)...")
        
        vector_store.add_documents(documents=batch)
        
        if current_batch < total_batches:
            time.sleep(0.4)
            
    print("✅ Đã lưu toàn bộ vector vào ChromaDB thành công!")
