import os
import time
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from utils.config import get_config


def embed_and_store(chunks: list, session_id: str, file_id: str):
    """Nhúng và lưu vào ChromaDB với metadata session_id và file_id."""
    chroma_path = get_config("CHROMA_DB_PATH", "./vector_store")

    print(f"⏳ Đang nhúng tổng cộng {len(chunks)} chunks bằng Cohere...")

    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vector_store = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
        collection_name="academic_chatbot"
    )

    # Gắn thêm metadata cho từng chunk để lọc theo session/file
    for doc in chunks:
        doc.metadata = {
            **(doc.metadata or {}),
            "session_id": session_id,
            "file_id": file_id,
            "filename": os.path.basename(doc.metadata.get("source", "")) if doc.metadata else None,
        }

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
