import os


def get_config(key: str, default: str | None = None) -> str | None:
    """Read config from environment first, then Streamlit secrets if available."""
    value = os.getenv(key)
    if value is not None and value != "":
        return value

    try:
        import streamlit as st

        if key in st.secrets:
            secret_value = st.secrets[key]
            if secret_value is not None:
                return str(secret_value)
    except Exception:
        pass

    return default
