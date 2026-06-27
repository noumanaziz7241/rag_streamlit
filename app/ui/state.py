"""Streamlit session state helpers."""

from __future__ import annotations

import streamlit as st

from memory_agent.config import DEFAULT_SESSION_ID, DEFAULT_USER_ID


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "chat_api": None,
        "user_id": DEFAULT_USER_ID,
        "active_session_id": DEFAULT_SESSION_ID,
        "messages": [],
        "history_session_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_messages_from_checkpoint() -> None:
    """Load chat messages from LangGraph for the active session."""
    if st.session_state.chat_api is None:
        return

    session_id = st.session_state.active_session_id
    if st.session_state.history_session_id == session_id:
        return

    st.session_state.messages = st.session_state.chat_api.get_history(
        user_id=st.session_state.user_id,
        session_id=session_id,
    )
    st.session_state.history_session_id = session_id
