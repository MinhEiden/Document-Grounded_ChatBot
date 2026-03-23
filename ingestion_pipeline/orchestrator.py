import uuid
from .loader import load_single_document
from .chunker import chunk_documents
from .embedder import embed_and_store


def ingest_file(file_path: str, session_id: str) -> dict | None:
    """Ingest một file đơn lẻ, gắn session_id và trả về thông tin ingest."""
    documents = load_single_document(file_path)
    if not documents:
        return None

    chunks = chunk_documents(documents)
    if not chunks:
        return None

    file_id = uuid.uuid4().hex
    embed_and_store(chunks, session_id=session_id, file_id=file_id)

    return {
        "file_id": file_id,
        "chunk_count": len(chunks),
    }
