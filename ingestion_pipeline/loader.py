import os
import unicodedata
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_documents(directory: str) -> list:
    """Đọc file bằng PyMuPDFLoader (PDF) và Docling (DOCX), chuẩn hóa NFC cho tiếng Việt."""
    documents = []
    converter = DocumentConverter()
    
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            filepath = os.path.join(root, filename)
            
            try:
                if ext == ".pdf":
                    print(f"📄 Đang đọc bằng PyMuPDFLoader: {filepath}")
                    loader = PyMuPDFLoader(filepath)
                    loaded_docs = loader.load()
                    text_content = "\n\n".join([d.page_content for d in loaded_docs])
                else:  
                    print(f"📄 Đang đọc bằng Docling: {filepath}")
                    result = converter.convert(filepath)
                    text_content = result.document.export_to_markdown()
                
                normalized_text = unicodedata.normalize('NFC', text_content)
                
                doc = Document(
                    page_content=normalized_text,
                    metadata={"source": filename, "type": ext}
                )
                documents.append(doc)
            except Exception as e:
                print(f"❌ Lỗi đọc file {filepath}: {e}")
                
    print(f"✅ Đã đọc {len(documents)} tài liệu từ thư mục '{directory}'")
    return documents
