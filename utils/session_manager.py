import uuid


def get_or_create_session_id(state) -> str:
    """Lấy hoặc tạo session_id và lưu trong Streamlit session_state."""
    if "session_id" not in state:
        state.session_id = uuid.uuid4().hex
    return state.session_id
