"""Streamlit application entry point."""

from __future__ import annotations

# Streamlit executes this file with `app/` on sys.path — fix before package imports.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.bootstrap import bootstrap
from app.ui.chat import render_chat
from app.ui.config_setup import ensure_configured, render_config_setup
from app.ui.sidebar import render_sidebar
from app.ui.state import initialize_session_state, sync_messages_from_checkpoint
from memory_agent.api import ChatAPI
from memory_agent.config import auto_index_sample_corpus_enabled, DEFAULT_DB_PATH


def main() -> None:
    """Run the Streamlit chat application."""
    bootstrap()

    st.set_page_config(
        page_title="Memory Agent Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not ensure_configured():
        render_config_setup()
        return

    initialize_session_state()

    if st.session_state.chat_api is None:
        with st.spinner("Initializing chat agent..."):
            st.session_state.chat_api = ChatAPI(db_path=DEFAULT_DB_PATH)

    if auto_index_sample_corpus_enabled() and not st.session_state.get("sample_corpus_bootstrapped"):
        if not st.session_state.chat_api.sample_corpus_is_indexed():
            with st.spinner("Indexing sample corpus for demo…"):
                try:
                    st.session_state.chat_api.index_sample_corpus()
                    st.session_state.sample_corpus_index_error = None
                except Exception as exc:
                    st.session_state.sample_corpus_index_error = str(exc)
        st.session_state.sample_corpus_bootstrapped = True

    render_sidebar()
    sync_messages_from_checkpoint()
    render_chat()


if __name__ == "__main__":
    main()
