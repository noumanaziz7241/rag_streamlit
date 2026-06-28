"""Configuration setup UI when API keys are missing."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from memory_agent.config import PROJECT_ROOT, get_missing_config_keys
from memory_agent.google.credentials import (
    DEFAULT_CREDENTIALS_FILENAME,
    DEFAULT_TOKEN_FILENAME,
    get_credentials_path,
    get_google_auth_error,
    get_token_path,
)


def render_config_setup() -> None:
    """Show instructions for configuring API keys and secrets."""
    missing = get_missing_config_keys()
    google_error = get_google_auth_error()

    st.error("API configuration is incomplete. The app cannot start until required keys are set.")
    st.markdown("### Missing keys")
    for key in missing:
        st.markdown(f"- `{key}`")

    if google_error:
        st.warning(google_error)

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    env_path = PROJECT_ROOT / ".env"
    credentials_path = get_credentials_path()
    token_path = get_token_path()

    st.markdown("### Recommended — Google AI Studio API key")
    st.markdown(
        "If your GCP org blocks **service account key creation** or OAuth is **internal-only**, "
        "use a [Google AI Studio](https://aistudio.google.com/apikey) key — no Vertex or OAuth required."
    )

    st.markdown("### Streamlit Cloud — paste JSON in Secrets")
    st.markdown(
        "Open **App settings → Secrets** on [share.streamlit.io](https://share.streamlit.io). "
        "Set `GEMINI_API_KEY`, or use Vertex via `[google_service_account]` / "
        "`[google_client_secret.web]` + `[google_token]`. "
        "See [docs/DEPLOY.md](docs/DEPLOY.md) for the full template."
    )

    st.markdown("### Local — Google credentials JSON (Vertex)")
    st.markdown(
        f"Place `{DEFAULT_CREDENTIALS_FILENAME}` in the project root, then run "
        f"`python scripts/google_auth.py` to create `{DEFAULT_TOKEN_FILENAME}`. "
        "Or set `GEMINI_API_KEY` in `.env` — it is used automatically when OAuth/Vertex is incomplete."
    )
    if credentials_path:
        st.success(f"Found credentials file: `{credentials_path}`")
    if token_path:
        st.success(f"Found OAuth token: `{token_path}`")

    st.markdown("### Pinecone secrets example")
    st.code(
        f"""# {secrets_path}
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "memory-agent-domain"
PINECONE_MEMORY_INDEX_NAME = "memory-agent-memory"

# Google — pick one (see docs/DEPLOY.md)
GEMINI_API_KEY = "your-gemini-api-key"
# [google_service_account]
# type = "service_account"
# project_id = "..."
""",
        language="toml",
    )

    st.markdown("### Local `.env` fallback")
    st.code(
        f"""# {env_path}
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=memory-agent-domain
PINECONE_MEMORY_INDEX_NAME=memory-agent-memory
GEMINI_API_KEY=your-gemini-api-key
""",
        language="bash",
    )

    st.info(
        "Gemini is used for chat, embeddings, and multimodal understanding. "
        "DeepSeek fallback is commented out until an API key is available."
    )

    if st.button("I added the keys — reload", use_container_width=True):
        st.rerun()


def ensure_configured() -> bool:
    """Return True when all required configuration is available."""
    return not get_missing_config_keys()
