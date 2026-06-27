"""Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from app.bootstrap import bootstrap
from app.ui.chat import render_chat
from app.ui.sidebar import render_sidebar
from app.ui.state import initialize_session_state, sync_messages_from_checkpoint
from memory_agent.api import ChatAPI
from memory_agent.config import DEFAULT_DB_PATH


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

    initialize_session_state()

    if st.session_state.chat_api is None:
        with st.spinner("Initializing chat agent..."):
            st.session_state.chat_api = ChatAPI(db_path=DEFAULT_DB_PATH)

    render_sidebar()
    sync_messages_from_checkpoint()
    render_chat()


if __name__ == "__main__":
    main()
