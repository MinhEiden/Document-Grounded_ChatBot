import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def _convert_file(filepath: str) -> Document | None:
    """Đọc một file đơn bằng Docling và trả về Document; trả None nếu lỗi."""
    converter = DocumentConverter()
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None

    print(f"📄 Đang đọc bằng Docling: {filepath}")
    try:
        result = converter.convert(filepath)
        text_content = result.document.export_to_markdown()
        return Document(
            page_content=text_content,
            metadata={"source": filepath, "type": ext}
        )
    except Exception as e:
        print(f"❌ Lỗi đọc file {filepath}: {e}")
        return None


def load_documents(directory: str) -> list:
    """Đọc tất cả file hợp lệ trong thư mục (giữ để tương thích)."""
    documents = []
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            filepath = os.path.join(root, filename)
            doc = _convert_file(filepath)
            if doc:
                documents.append(doc)
    print(f"✅ Đã đọc {len(documents)} tài liệu từ thư mục '{directory}'")
    return documents


def load_single_document(filepath: str) -> list:
    """Đọc một file đơn, trả về list để tái sử dụng với chunker/embedder."""
    doc = _convert_file(filepath)
    return [doc] if doc else []
