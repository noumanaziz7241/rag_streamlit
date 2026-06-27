"""Configuration setup UI when API keys are missing."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from memory_agent.config import PROJECT_ROOT, REQUIRED_CONFIG_KEYS, get_missing_config_keys


def render_config_setup() -> None:
    """Show instructions for configuring API keys and secrets."""
    missing = get_missing_config_keys()

    st.error("API configuration is incomplete. The app cannot start until required keys are set.")
    st.markdown("### Missing keys")
    for key in missing:
        st.markdown(f"- `{key}`")

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    env_path = PROJECT_ROOT / ".env"

    st.markdown("### Option 1 — Streamlit secrets (recommended)")
    st.code(
        f"""# {secrets_path}
GEMINI_API_KEY = "your-gemini-api-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-domain-index"
PINECONE_MEMORY_INDEX_NAME = "your-memory-index"
""",
        language="toml",
    )

    st.markdown("### Option 2 — `.env` file in the project root")
    st.code(
        f"""# {env_path}
GEMINI_API_KEY=your-gemini-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-domain-index
PINECONE_MEMORY_INDEX_NAME=your-memory-index
""",
        language="bash",
    )

    st.info(
        "`GOOGLE_API_KEY` is also accepted as an alias for `GEMINI_API_KEY`. "
        "After saving your keys, refresh this page."
    )

    if st.button("I added the keys — reload", use_container_width=True):
        st.rerun()


def ensure_configured() -> bool:
    """Return True when all required configuration is available."""
    return not get_missing_config_keys()
