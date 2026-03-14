import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_documents(directory: str) -> list:
    """Đọc file bằng Docling (giữ nguyên layout markdown, bảng biểu tốt hơn)."""
    documents = []
    converter = DocumentConverter()
    
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            filepath = os.path.join(root, filename)
            print(f"📄 Đang đọc bằng Docling: {filepath}")
            
            try:
                # Convert document sang Docling format
                result = converter.convert(filepath)
                # Trích xuất sang Markdown giúp Semantic Chunker hiểu tốt cấu trúc
                text_content = result.document.export_to_markdown()
                
                doc = Document(
                    page_content=text_content,
                    metadata={"source": filepath, "type": ext}
                )
                documents.append(doc)
            except Exception as e:
                print(f"❌ Lỗi đọc file {filepath}: {e}")
                
    print(f"✅ Đã đọc {len(documents)} tài liệu từ thư mục '{directory}'")
    return documents
