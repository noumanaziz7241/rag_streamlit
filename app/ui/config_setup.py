"""Configuration setup UI when API keys are missing."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from memory_agent.config import PROJECT_ROOT, get_missing_config_keys
from memory_agent.config_diagnostics import format_key_status, get_config_diagnostics
from memory_agent.google.credentials import (
    DEFAULT_CREDENTIALS_FILENAME,
    DEFAULT_TOKEN_FILENAME,
    get_credentials_path,
    get_token_path,
)

MINIMAL_STREAMLIT_SECRETS = """\
PINECONE_API_KEY = "pcsk_..."
PINECONE_INDEX_NAME = "memory-agent-domain"
PINECONE_MEMORY_INDEX_NAME = "memory-agent-memory"

# Get from https://aistudio.google.com/apikey
GEMINI_API_KEY = "AIza..."

AUTO_INDEX_SAMPLE_CORPUS = true
"""


def render_config_setup() -> None:
    """Show instructions for configuring API keys and secrets."""
    missing = get_missing_config_keys()
    diagnostics = get_config_diagnostics()

    st.error("API configuration is incomplete. The app cannot start until required keys are set.")
    st.markdown("### Missing keys")
    for key in missing:
        st.markdown(f"- `{key}`")

    st.markdown("### Secrets diagnostic")
    if diagnostics.secrets_load_error:
        st.error(f"Could not load Streamlit secrets: {diagnostics.secrets_load_error}")
    elif diagnostics.secrets_available:
        st.success("Streamlit secrets file loaded.")
    else:
        st.warning("No Streamlit secrets detected (normal for local dev without secrets.toml).")

    for key, status in diagnostics.keys.items():
        st.markdown(f"- `{key}`: {format_key_status(status)}")

    if diagnostics.google_auth_ok:
        st.success(f"Google auth: OK (`{diagnostics.google_auth_mode}`)")
    elif diagnostics.google_error:
        st.warning(diagnostics.google_error)

    for hint in diagnostics.hints:
        st.info(hint)

    st.markdown("### Minimal Streamlit Cloud secrets (copy this)")
    st.markdown(
        "In [share.streamlit.io](https://share.streamlit.io) → your app → "
        "**Settings → Secrets**, paste **only** this (replace placeholder values). "
        "Use **TOML**, not JSON. Do **not** leave OAuth sections from the example unless you use Vertex."
    )
    st.code(MINIMAL_STREAMLIT_SECRETS, language="toml")

    st.markdown("### After saving secrets")
    st.markdown(
        "1. Click **Save**\n"
        "2. Open **Manage app → Reboot app** (required — secrets are not picked up until reboot)\n"
        "3. Confirm main file is `app/main.py`"
    )

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    env_path = PROJECT_ROOT / ".env"
    credentials_path = get_credentials_path()
    token_path = get_token_path()

    with st.expander("Advanced — Vertex AI / local setup"):
        st.markdown(
            "Full templates: [docs/DEPLOY.md](docs/DEPLOY.md). "
            f"Local OAuth: `{DEFAULT_CREDENTIALS_FILENAME}` + `{DEFAULT_TOKEN_FILENAME}`."
        )
        if credentials_path:
            st.success(f"Found credentials file: `{credentials_path}`")
        if token_path:
            st.success(f"Found OAuth token: `{token_path}`")
        st.code(
            f"""# Local {secrets_path}
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "memory-agent-domain"
PINECONE_MEMORY_INDEX_NAME = "memory-agent-memory"
GEMINI_API_KEY = "your-gemini-api-key"
""",
            language="toml",
        )
        st.code(
            f"""# Local {env_path}
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=memory-agent-domain
PINECONE_MEMORY_INDEX_NAME=memory-agent-memory
GEMINI_API_KEY=your-gemini-api-key
""",
            language="bash",
        )

    if st.button("I added the keys — reload", use_container_width=True):
        st.rerun()


def ensure_configured() -> bool:
    """Return True when all required configuration is available."""
    return not get_missing_config_keys()
