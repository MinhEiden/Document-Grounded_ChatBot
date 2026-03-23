import os
import uuid
from pathlib import Path


def save_uploaded_file(uploaded_file, session_id: str, base_dir: str = "temp_uploads") -> str:
    """Lưu file upload vào thư mục session, trả về đường dẫn đã lưu."""
    safe_name = Path(uploaded_file.name).name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"

    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / unique_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)
