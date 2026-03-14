from dotenv import load_dotenv

# Force reload dotenv để đảm bảo lấy đúng API key hiện tại từ file .env
load_dotenv(override=True)

import os
import shutil
from .loader import load_documents
from .chunker import chunk_documents
from .embedder import embed_and_store

def run_ingestion_pipeline(data_path: str = "./TrainData"):
    """
    Thực thi toàn bộ Ingestion Pipeline:
    1. Xóa toàn bộ DB cũ nếu tồn tại trong chroma_db.
    2. Load tài liệu (pdf, docx).
    3. Trích xuất Semantic Chunks.
    4. Nhúng và lưu vector vào ChromaDB.
    """
    # Nạp biến môi trường từ .env (cho khóa API của Cohere)
    load_dotenv()
    
    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    print(f"Kiểm tra dữ liệu cũ trong: {chroma_path}")
    if os.path.exists(chroma_path):
        import glob
        # Kiểm tra xem có file nào trong chroma_db không
        files = glob.glob(os.path.join(chroma_path, "*"))
        if files:
            print("🗑️ Phát hiện database cũ, đang tiến hành xóa toàn bộ...")
            shutil.rmtree(chroma_path)
            os.makedirs(chroma_path, exist_ok=True)
            print("✅ Đã làm sạch ChromaDB cũ.")
    else:
        os.makedirs(chroma_path, exist_ok=True)

    print(f"\n🚀 Bắt đầu Ingestion Pipeline cho thư mục: {data_path}")
    documents = load_documents(data_path)
    if not documents:
        print("❌ Không tìm thấy tài liệu phù hợp nào.")
        return

    chunks = chunk_documents(documents)
    if not chunks:
        print("❌ Thất bại trong việc chia chunk tài liệu.")
        return

    embed_and_store(chunks)
    print("\n🎉 Hoàn thành toàn bộ quy trình Ingestion Pipeline!")

