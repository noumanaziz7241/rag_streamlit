"""Application configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the repository root regardless of Streamlit's working directory.
load_dotenv(PROJECT_ROOT / ".env")

NAMESPACE = "polaris"
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "main_session"
DEFAULT_DB_PATH = os.getenv("CHAT_DB_PATH", str(PROJECT_ROOT / "chat_memory.db"))
UPLOADS_DIR = str(PROJECT_ROOT / "data" / "uploads")

# Latest Gemini embedding model — natively multimodal (text, image, video, audio, PDF).
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768
GEMINI_MULTIMODAL_MODEL = "gemini-2.5-flash"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"

# DeepSeek chat fallback — disabled until an API key is available.
# Set to True and uncomment the block in memory_agent/google/chat_model.py.
DEEPSEEK_ENABLED = False

PDF_PAGES_PER_CHUNK = 6
MAX_VIDEO_SECONDS = 120
MAX_AUDIO_SECONDS = 180
MAX_IMAGES_PER_EMBED_REQUEST = 6

# Primary key -> acceptable aliases (env vars and Streamlit secrets).
CONFIG_ALIASES: dict[str, list[str]] = {
    "GEMINI_API_KEY": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "DEEPSEEK_API_KEY": ["DEEPSEEK_API_KEY"],
    "PINECONE_API_KEY": ["PINECONE_API_KEY"],
    "PINECONE_INDEX_NAME": ["PINECONE_INDEX_NAME"],
    "PINECONE_MEMORY_INDEX_NAME": ["PINECONE_MEMORY_INDEX_NAME"],
    "GEMINI_MULTIMODAL_MODEL": ["GEMINI_MULTIMODAL_MODEL"],
    "GEMINI_CHAT_MODEL": ["GEMINI_CHAT_MODEL"],
    "GOOGLE_CREDENTIALS_PATH": ["GOOGLE_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"],
    "GOOGLE_VERTEX_LOCATION": ["GOOGLE_VERTEX_LOCATION"],
}


def _read_streamlit_secret(key: str) -> str | None:
    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            value = st.secrets[key]
            if value is not None and str(value).strip():
                return str(value).strip()
    except Exception:
        pass
    return None


def get_config_value(key: str) -> str:
    """Read configuration from Streamlit secrets, then environment variables."""
    candidates = CONFIG_ALIASES.get(key, [key])

    for candidate in candidates:
        secret_value = _read_streamlit_secret(candidate)
        if secret_value:
            return secret_value

    for candidate in candidates:
        env_value = os.getenv(candidate)
        if env_value and env_value.strip():
            return env_value.strip()

    aliases = ", ".join(candidates)
    raise ValueError(
        f"Missing required configuration: {key}. "
        f"Set one of ({aliases}) in `.streamlit/secrets.toml` or a `.env` file "
        f"at `{PROJECT_ROOT}`."
    )


def has_deepseek_api_key() -> bool:
    try:
        get_config_value("DEEPSEEK_API_KEY")
        return True
    except ValueError:
        return False


def has_google_auth() -> bool:
    from memory_agent.google.credentials import has_google_auth as _has_google_auth

    return _has_google_auth()


def get_missing_config_keys() -> list[str]:
    """Return required config keys that are not currently set."""
    missing: list[str] = []

    if not has_google_auth():
        missing.append("GOOGLE_CREDENTIALS (Streamlit secrets, JSON file, or GEMINI_API_KEY)")

    # DeepSeek optional while DEEPSEEK_ENABLED is False:
    # if DEEPSEEK_ENABLED and not has_deepseek_api_key():
    #     missing.append("DEEPSEEK_API_KEY")

    for key in ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_MEMORY_INDEX_NAME"):
        try:
            get_config_value(key)
        except ValueError:
            missing.append(key)

    return missing


def get_gemini_multimodal_model() -> str:
    """Resolve multimodal model from config with a stable default."""
    try:
        return get_config_value("GEMINI_MULTIMODAL_MODEL")
    except ValueError:
        return GEMINI_MULTIMODAL_MODEL


def get_gemini_chat_model() -> str:
    """Resolve chat model from config with a stable default."""
    try:
        return get_config_value("GEMINI_CHAT_MODEL")
    except ValueError:
        return GEMINI_CHAT_MODEL
