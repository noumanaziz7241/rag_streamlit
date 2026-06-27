"""Application configuration and constants."""

from __future__ import annotations

import os

NAMESPACE = "polaris"
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "main_session"
DEFAULT_DB_PATH = "chat_memory.db"


def get_config_value(key: str) -> str:
    """Read configuration from Streamlit secrets with .env fallback."""
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except (FileNotFoundError, RuntimeError, ImportError):
        pass

    value = os.getenv(key)
    if value:
        return value
    raise ValueError(f"Missing required configuration: {key}")
