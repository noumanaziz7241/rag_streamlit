"""Application configuration and constants."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from the repository root regardless of Streamlit's working directory.
load_dotenv(PROJECT_ROOT / ".env")


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud."""
    runtime = os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower()
    return runtime in {"cloud", "streamlit-cloud", "space"}


def _path_is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def resolve_db_path() -> str:
    """Pick a writable SQLite path (Streamlit Cloud cannot write to a read-only repo mount)."""
    explicit = os.getenv("CHAT_DB_PATH")
    if explicit and explicit.strip():
        return explicit.strip()

    preferred = PROJECT_ROOT / "chat_memory.db"
    if _path_is_writable(preferred.parent):
        return str(preferred)

    fallback = Path("/tmp/memory_agent/chat_memory.db")
    _path_is_writable(fallback.parent)
    return str(fallback)


def resolve_uploads_dir() -> str:
    """Pick a writable uploads directory for media chunks."""
    explicit = os.getenv("UPLOADS_DIR")
    if explicit and explicit.strip():
        return explicit.strip()

    preferred = PROJECT_ROOT / "data" / "uploads"
    if _path_is_writable(preferred):
        return str(preferred)

    fallback = Path("/tmp/memory_agent/uploads")
    _path_is_writable(fallback)
    return str(fallback)


NAMESPACE = "polaris"
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "main_session"
DEFAULT_DB_PATH = resolve_db_path()
UPLOADS_DIR = resolve_uploads_dir()
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


AUTO_INDEX_SAMPLE_CORPUS = _env_flag("AUTO_INDEX_SAMPLE_CORPUS")


def auto_index_sample_corpus_enabled() -> bool:
    """Default on for Streamlit Cloud so the live demo works without manual indexing."""
    raw = os.getenv("AUTO_INDEX_SAMPLE_CORPUS")
    if raw is not None and str(raw).strip():
        return _env_flag("AUTO_INDEX_SAMPLE_CORPUS")
    return is_streamlit_cloud()

# Latest Gemini embedding model — natively multimodal (text, image, video, audio, PDF).
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768
GEMINI_MULTIMODAL_MODEL = "gemini-2.5-flash"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
# Used when the primary model returns 503 / capacity errors.
DEFAULT_GEMINI_MODEL_FALLBACKS = ("gemini-2.0-flash", "gemini-2.0-flash-lite")

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
    "GEMINI_MODEL_FALLBACKS": ["GEMINI_MODEL_FALLBACKS"],
    "GOOGLE_CREDENTIALS_PATH": ["GOOGLE_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"],
    "GOOGLE_VERTEX_LOCATION": ["GOOGLE_VERTEX_LOCATION"],
}


def _read_streamlit_secret(key: str) -> str | None:
    try:
        import streamlit as st

        if not hasattr(st, "secrets"):
            return None

        try:
            if key not in st.secrets:
                return None
            value = st.secrets[key]
        except (KeyError, TypeError):
            return None
        except Exception as exc:
            # Invalid secrets TOML on Streamlit Cloud surfaces on access.
            if "Secrets" in str(exc) or "TOML" in str(exc):
                raise
            return None

        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:
        return None


def read_secret_optional(key: str) -> str | None:
    """Read a secret from Streamlit or environment without raising."""
    for candidate in CONFIG_ALIASES.get(key, [key]):
        secret_value = _read_streamlit_secret(candidate)
        if secret_value:
            return secret_value
        env_value = os.getenv(candidate)
        if env_value and env_value.strip():
            return env_value.strip()
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


def get_gemini_model_fallbacks() -> list[str]:
    """Secondary Gemini models when the primary hits capacity limits."""
    try:
        raw = get_config_value("GEMINI_MODEL_FALLBACKS")
    except ValueError:
        raw = os.getenv("GEMINI_MODEL_FALLBACKS", "")

    if raw and str(raw).strip():
        return [part.strip() for part in str(raw).split(",") if part.strip()]
    return list(DEFAULT_GEMINI_MODEL_FALLBACKS)


def get_gemini_model_chain(primary: str) -> list[str]:
    """Primary model first, then configured fallbacks (deduped)."""
    chain = [primary]
    for model in get_gemini_model_fallbacks():
        if model not in chain:
            chain.append(model)
    return chain
